"""B7: Agent-noV baseline (Agent without Verification).

This is an ablation baseline that uses the full CL-LLMOps pipeline
but removes the CAEVN + FIS verification stage. All hypotheses are
passed directly to action generation without gating.

This isolates the contribution of the verification mechanism.
"""

import time
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class AgentNoVerifyPipeline:
    """CL-LLMOps pipeline without verification (ablation baseline B7).

    Stages: Alarm -> Hypothesis -> [skip verification] -> Action -> Regression
    """

    def __init__(self, alarm_classifier, hypothesis_generator,
                 action_generator, regression_tester):
        self.alarm = alarm_classifier
        self.hypothesis = hypothesis_generator
        self.action = action_generator
        self.regression = regression_tester

    def run(self, signal, run_regression=True):
        """Execute pipeline without verification stage."""
        result = {"timings": {}}
        t0 = time.time()

        # Stage 1: Alarm
        t1 = time.time()
        alarm_result = self.alarm.classify(signal)
        result["alarm"] = alarm_result
        result["timings"]["alarm_ms"] = (time.time() - t1) * 1000

        # Stage 2: Hypothesis (single sample, no self-consistency)
        t2 = time.time()
        hyp_result = self.hypothesis.generate(alarm_result)
        result["hypothesis"] = {
            "selected": hyp_result["selected"],
            "consistency_score": hyp_result["consistency_score"],
        }
        result["timings"]["hypothesis_ms"] = (time.time() - t2) * 1000

        # Stage 3: SKIP VERIFICATION - always proceed
        verification_result = {
            "score": 1.0,
            "decision": "proceed",
            "alpha": 1.0,
            "consistency": hyp_result["consistency_score"],
        }
        result["verification"] = verification_result

        # Stage 4: Action
        t4 = time.time()
        action_result = self.action.generate(
            alarm_result, hyp_result, verification_result)
        result["action"] = action_result
        result["timings"]["action_ms"] = (time.time() - t4) * 1000

        # Stage 5: Regression
        if run_regression:
            t5 = time.time()
            reg_result = self.regression.run(
                lambda x: self.alarm.classify(x))
            result["regression"] = reg_result
            result["timings"]["regression_ms"] = (time.time() - t5) * 1000
        else:
            result["regression"] = {"passed": None, "skipped": True}

        result["timings"]["total_ms"] = (time.time() - t0) * 1000
        result["final_decision"] = "proceed"  # always proceed
        return result

    def batch_run(self, signals, run_regression=False):
        """Batch run without verification."""
        results = [self.run(s, run_regression=run_regression) for s in signals]
        avg_time = np.mean([r["timings"]["total_ms"] for r in results])
        return results, {
            "n_samples": len(results),
            "proceed_rate": 1.0,
            "defer_rate": 0.0,
            "reject_rate": 0.0,
            "avg_time_ms": avg_time,
        }
