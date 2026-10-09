"""Convert CARE-to-Compare (Zenodo 15846963, v6) raw datasets into the CSV layout
expected by experiments/data/care_dataset.py.

Usage:
    python experiments/data/prepare_care.py --care-root <extracted>/CARE_To_Compare


Rules:
- Emit ONLY rows with train_test == "prediction" (verified: one contiguous end
  block per dataset; the raw 1-year training rows are excluded for feasibility).
- Row label = 1 iff event_start <= time_stamp <= event_end for anomaly events
  (event_info.csv); normal datasets are all 0.
- Sensor channels: 10-minute AVERAGE channels only (Zenodo known-issues note
  recommends avg-only). Per farm, rank channels by non-zero fraction over the
  farm's full raw data (train+prediction), take top 52 (farm A has 54 avg
  channels, two of which - sensor_46, sensor_49 - are 100% zero and dropped).
  Ties broken by original column order. Original column names kept.
- NaN: linear interpolation over the emitted row order (both directions),
  remaining NaN filled with 0 (mirrors the 0-fill convention of farms B/C).
- Output: one CSV per dataset, comma-separated, columns:
  timestamp,<52 sensor columns>,label
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--care-root", type=Path, required=True,
                help="extracted Zenodo archive folder that contains 'Wind Farm A/B/C'")
ap.add_argument("--out", type=Path, default=HERE / "raw" / "care_wind")
ap.add_argument("--manifest", type=Path, default=HERE / "care_conversion_manifest.json")
args = ap.parse_args()
ROOT, OUT = args.care_root, args.out
OUT.mkdir(parents=True, exist_ok=True)
N_TAGS = 52

manifest = {"n_tags": N_TAGS, "farms": {}, "files": []}

for farm, letter in [("Wind Farm A", "A"), ("Wind Farm B", "B"), ("Wind Farm C", "C")]:
    fdir = ROOT / farm
    ei = pd.read_csv(fdir / "event_info.csv", sep=";").set_index("event_id")
    fd = pd.read_csv(fdir / "feature_description.csv", sep=";")
    ds = sorted((fdir / "datasets").glob("*.csv"), key=lambda p: int(p.stem))
    hdr = list(pd.read_csv(ds[0], sep=";", nrows=1).columns)
    avg_cols = [c for c in hdr if c.endswith("_avg")]
    bare_avg = [c for c in hdr if c in set(fd.loc[fd["statistics_type"] == "average", "sensor_name"])
                and not c.endswith("_avg")]
    cand = [c for c in hdr if c in set(avg_cols + bare_avg)]  # original file order
    # non-zero fraction over full raw data of this farm
    nz = pd.Series(0.0, index=cand); ntot = 0
    for f in ds:
        v = pd.read_csv(f, sep=";", usecols=cand)
        nz += (v.fillna(0) != 0).sum(); ntot += len(v)
    frac = nz / ntot
    order = sorted(range(len(cand)), key=lambda i: (-frac.iloc[i], i))
    sel_idx = sorted(order[:N_TAGS])
    selected = [cand[i] for i in sel_idx]
    manifest["farms"][letter] = {
        "n_candidate_avg_channels": len(cand),
        "selected_channels": selected,
        "selected_nonzero_frac": {c: round(float(frac[c]), 4) for c in selected},
        "dropped_channels": {cand[i]: round(float(frac.iloc[i]), 4) for i in order[N_TAGS:]},
    }
    for f in ds:
        eid = int(f.stem)
        r = ei.loc[eid]
        d = pd.read_csv(f, sep=";", usecols=["time_stamp", "train_test"] + selected)
        d = d[d["train_test"] == "prediction"].reset_index(drop=True)
        ts = pd.to_datetime(d["time_stamp"])
        if r["event_label"] == "anomaly":
            es, ee = pd.Timestamp(r["event_start"]), pd.Timestamp(r["event_end"])
            label = ((ts >= es) & (ts <= ee)).astype(int)
        else:
            label = pd.Series(0, index=d.index, dtype=int)
        vals = d[selected].astype(np.float64)
        n_nan_before = int(vals.isna().sum().sum())
        vals = vals.interpolate(method="linear", limit_direction="both").fillna(0.0)
        out = pd.concat([d["time_stamp"].rename("timestamp"), vals, label.rename("label")], axis=1)
        oname = f"care_{letter}_{eid}.csv"
        out.to_csv(OUT / oname, index=False)
        manifest["files"].append({
            "file": oname, "source": f"{farm}/datasets/{f.name}", "farm": letter, "event_id": eid,
            "event_label": r["event_label"], "rows": len(out),
            "label1_rows": int(label.sum()), "nan_cells_filled": n_nan_before,
            "ts_first": str(d["time_stamp"].iloc[0]), "ts_last": str(d["time_stamp"].iloc[-1]),
        })

manifest["total_rows"] = sum(x["rows"] for x in manifest["files"])
manifest["total_label1_rows"] = sum(x["label1_rows"] for x in manifest["files"])
manifest["n_files"] = len(manifest["files"])
with open(args.manifest, "w") as fh:
    json.dump(manifest, fh, indent=1)
print("files:", manifest["n_files"], "rows:", manifest["total_rows"],
      "label1_rows:", manifest["total_label1_rows"])
est = sum(max(0, x["rows"] - 144) for x in manifest["files"])
print("estimated windows (rows-144 per file):", est)
