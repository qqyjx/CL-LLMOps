#!/usr/bin/env bash
# Fetch the CWRU 12 kHz drive-end subset for ALL four motor loads (0-3 hp)
# and rename to loader-compatible class names. The loader
# (experiments/data/cwru_dataset.py::_filename_to_label) needs "normal" or
# fault-type+size tokens in the filename; raw CWRU numbers are dropped to
# avoid the "97"-"100" normal-detection trap (e.g. 197.mat contains "97").
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)/raw/cwru"
mkdir -p "$DIR"
cd "$DIR"

declare -A FILES=(
  ["Normal_0hp.mat"]="97"  ["Normal_1hp.mat"]="98"  ["Normal_2hp.mat"]="99"  ["Normal_3hp.mat"]="100"
  ["IR007_0hp.mat"]="105"  ["IR007_1hp.mat"]="106"  ["IR007_2hp.mat"]="107"  ["IR007_3hp.mat"]="108"
  ["IR014_0hp.mat"]="169"  ["IR014_1hp.mat"]="170"  ["IR014_2hp.mat"]="171"  ["IR014_3hp.mat"]="172"
  ["IR021_0hp.mat"]="209"  ["IR021_1hp.mat"]="210"  ["IR021_2hp.mat"]="211"  ["IR021_3hp.mat"]="212"
  ["OR007_0hp.mat"]="130"  ["OR007_1hp.mat"]="131"  ["OR007_2hp.mat"]="132"  ["OR007_3hp.mat"]="133"
  ["OR014_0hp.mat"]="197"  ["OR014_1hp.mat"]="198"  ["OR014_2hp.mat"]="199"  ["OR014_3hp.mat"]="200"
  ["OR021_0hp.mat"]="234"  ["OR021_1hp.mat"]="235"  ["OR021_2hp.mat"]="236"  ["OR021_3hp.mat"]="237"
  ["B007_0hp.mat"]="118"   ["B007_1hp.mat"]="119"   ["B007_2hp.mat"]="120"   ["B007_3hp.mat"]="121"
  ["B014_0hp.mat"]="185"   ["B014_1hp.mat"]="186"   ["B014_2hp.mat"]="187"   ["B014_3hp.mat"]="188"
  ["B021_0hp.mat"]="222"   ["B021_1hp.mat"]="223"   ["B021_2hp.mat"]="224"   ["B021_3hp.mat"]="225"
)

fails=0
for f in $(printf '%s\n' "${!FILES[@]}" | sort); do
  n="${FILES[$f]}"
  if [[ -f "$f" ]]; then echo "[skip] $f"; continue; fi
  echo "[get] $f <- ${n}.mat"
  if ! curl -fsSL --retry 3 --connect-timeout 20 --max-time 300 \
      "https://engineering.case.edu/sites/default/files/${n}.mat" -o "$f"; then
    echo "FAIL $f"; fails=$((fails+1))
  fi
done
echo "CWRU_DOWNLOAD_DONE fails=${fails}"
ls -lh "$DIR" | tail -n +2
(cd "$DIR" && sha256sum -c ../../cwru_CHECKSUMS.txt) || echo "checksum mismatch: compare with experiments/data/cwru_CHECKSUMS.txt"
