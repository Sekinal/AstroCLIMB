"""Aggregate matched gold-only vs public-initialized training logs; no data access.

Reads retrieved train.jsonl logs (as written by scripts/vlm_cxi_peft.py) and
writes a public aggregate receipt with hashes and validation scores only.
No private paths, row content, labels, or traces are echoed.

Example manifest (JSON file passed via --manifest):
  {"expected_horizon": 837, "runs": [
    {"arm": "gold_only", "seed": 7, "log": "/tmp/gold7/train.jsonl", "exitcode": 0},
    {"arm": "public4k", "seed": 7, "log": "/tmp/pub7/train.jsonl", "exitcode": 0}]}

Run:
  python analysis/summarize_matched_runs.py --manifest manifest.json --out receipt.json

Rules: blank JSONL lines are ignored; any other malformed line raises
ValueError with only the log file basename (no directories). Duplicate
(arm, seed) manifest entries raise ValueError. Each ready event is checked
against the manifest: config.seed must equal the manifest seed, top-level
ready train_rows must be 2230, top-level ready val_rows must be 794, val_every
first step (279 when horizon is 837; horizon//3 for other divisible
horizons), public4k requires a nonempty config.init_checkpoint and gold_only
requires init_checkpoint None; missing config values and mismatches are
reported as incomplete with a reason (or ValueError with a clear reason for
bad types). The expected validation steps are the fixed protocol thirds
([279, 558, 837] when horizon is 837), independent of val_every. Missing
validation steps give incomplete with no selected score; duplicated,
nonfinite, out of range, missing-field, or unexpected validation steps raise
ValueError. Each expected validation step requires a checkpoint with
reload_verified True; missing or unverified checkpoints give incomplete. Each
complete run must end with a last done (or already_complete) event whose step
equals expected_horizon and which appears after the final validation;
otherwise incomplete with no selected score. Among complete validation events
the greatest four-class macro F1 wins with ties broken to the earliest step;
final_epoch_score is the macro at exactly expected_horizon. Paired
differences use only seeds where both arms are complete. Descriptive means
are reported only when there are at least two pairs; no confidence intervals
are computed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ("gold_only", "public4k")
TRAIN_ROWS = 2230
VAL_ROWS = 794


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_jsonl(path):
    name = Path(path).name
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            if raw.strip() == "":
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{name}:{lineno}: malformed JSON: {e}") from e
            if not isinstance(obj, dict) or not isinstance(obj.get("event"), str):
                raise ValueError(f"{name}:{lineno}: each line must be an object with str 'event'")
            events.append(obj)
    return events


def expected_validation_steps(ready, horizon):
    del ready
    if horizon % 3 == 0:
        return [horizon // 3, 2 * horizon // 3, horizon]
    return [horizon]


def summarize_run(entry, horizon):
    if not isinstance(entry, dict):
        raise ValueError("each run entry must be an object")
    arm = entry.get("arm")
    seed = entry.get("seed")
    log = entry.get("log")
    exitcode = entry.get("exitcode")
    if arm not in ARMS:
        raise ValueError(f"run arm must be one of {list(ARMS)}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("run seed must be an int")
    if not isinstance(log, str) or not log:
        raise ValueError("run log must be a nonempty path string")
    if isinstance(exitcode, bool) or not isinstance(exitcode, int):
        raise ValueError("run exitcode must be an int")
    source_sha256 = sha256_file(log)

    def incomplete(reason):
        return {"arm": arm, "seed": seed, "status": "incomplete",
                "expected_horizon": horizon, "source_sha256": source_sha256,
                "reason": reason}

    if exitcode != 0:
        return incomplete(f"nonzero exitcode {exitcode}")
    events = parse_jsonl(log)
    readies = [e for e in events if e.get("event") == "ready"]
    if not readies:
        return incomplete("missing ready event")
    ready = readies[-1]
    total = ready.get("total_steps")
    if total != horizon:
        return incomplete(f"ready total_steps {total!r} != expected_horizon {horizon}")
    cfg = ready.get("config")
    if not isinstance(cfg, dict):
        return incomplete("missing ready.config object")
    expected = expected_validation_steps(ready, horizon)
    required_val_every = expected[0]
    val_every = cfg.get("val_every")
    if isinstance(val_every, bool) or not isinstance(val_every, int):
        return incomplete(f"missing ready.config.val_every; required {required_val_every}")
    if val_every != required_val_every:
        return incomplete(
            f"ready.config.val_every {val_every!r} != fixed protocol {required_val_every}")
    cfg_seed = cfg.get("seed")
    if cfg_seed != seed:
        return incomplete(f"ready.config.seed {cfg_seed!r} != manifest seed {seed!r}")
    if ready.get("train_rows") != TRAIN_ROWS:
        return incomplete(f"ready.train_rows {ready.get('train_rows')!r} != {TRAIN_ROWS}")
    if ready.get("val_rows") != VAL_ROWS:
        return incomplete(f"ready.val_rows {ready.get('val_rows')!r} != {VAL_ROWS}")
    init_ckpt = cfg.get("init_checkpoint")
    if arm == "public4k":
        if not isinstance(init_ckpt, str) or not init_ckpt:
            return incomplete("public4k requires nonempty ready.config.init_checkpoint")
    else:
        if init_ckpt is not None:
            return incomplete("gold_only requires ready.config.init_checkpoint None")
    indexed_validations = [(i, e) for i, e in enumerate(events) if e.get("event") == "validation"]
    steps = []
    for _, v in indexed_validations:
        if "step" not in v or "macro" not in v:
            raise ValueError("validation event missing 'step' or 'macro'")
        s = v["step"]
        m = v["macro"]
        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError("validation 'step' must be an int")
        if isinstance(m, bool) or not isinstance(m, (int, float)):
            raise ValueError("validation 'macro' must be a number")
        mf = float(m)
        if not math.isfinite(mf):
            raise ValueError("validation 'macro' must be finite")
        if mf < 0.0 or mf > 1.0:
            raise ValueError("validation 'macro' out of range [0, 1]")
        b = v.get("best")
        if b is not None:
            if isinstance(b, bool) or not isinstance(b, (int, float)):
                raise ValueError("validation 'best' must be a number")
            bf = float(b)
            if not math.isfinite(bf) or bf < 0.0 or bf > 1.0:
                raise ValueError("validation 'best' must be finite in [0, 1]")
        steps.append(s)
    if len(steps) != len(set(steps)):
        raise ValueError("duplicated validation steps")
    unexpected = sorted(s for s in steps if s not in expected)
    if unexpected:
        raise ValueError(f"unexpected validation steps {unexpected} != expected {sorted(expected)}")
    if sorted(steps) != sorted(expected):
        missing = sorted(s for s in expected if s not in steps)
        return incomplete(f"missing validation steps {missing}; expected {sorted(expected)}")
    scored = sorted(((float(v["macro"]), int(v["step"])) for _, v in indexed_validations),
                    key=lambda t: (-t[0], t[1]))
    checkpoints = [e for e in events if e.get("event") == "checkpoint"]
    verified_steps = set()
    for c in checkpoints:
        if c.get("reload_verified") is not True:
            return incomplete("checkpoint reload_verified is not True")
        cs = c.get("step")
        if cs is None:
            return incomplete("checkpoint missing 'step'")
        if isinstance(cs, bool) or not isinstance(cs, int):
            raise ValueError("checkpoint 'step' must be an int")
        verified_steps.add(cs)
    missing_ckpt = sorted(s for s in expected if s not in verified_steps)
    if missing_ckpt:
        return incomplete(f"missing verified checkpoint at steps {missing_ckpt}")
    finals = [(i, e) for i, e in enumerate(events) if e.get("event") in ("done", "already_complete")]
    if not finals:
        return incomplete("missing done event")
    for _, e in finals:
        s = e.get("step")
        if s is not None and (isinstance(s, bool) or not isinstance(s, int)):
            raise ValueError("done 'step' must be an int")
    last_idx, last = finals[-1]
    if last.get("step") != horizon:
        return incomplete(f"last done step {last.get('step')!r} != expected_horizon {horizon}")
    max_val_idx = max(i for i, _ in indexed_validations)
    if last_idx < max_val_idx:
        return incomplete("last done event is not after final validation")
    best_score, chosen_step = scored[0][0], scored[0][1]
    ordered = sorted(((int(v["step"]), float(v["macro"])) for _, v in indexed_validations),
                     key=lambda t: t[0])
    final_epoch_score = next(m for s, m in ordered if s == horizon)
    return {"arm": arm, "seed": seed, "status": "complete",
            "expected_horizon": horizon, "source_sha256": source_sha256,
            "validation_scores": [{"step": s, "macro_f1": m} for s, m in ordered],
            "chosen_step": chosen_step, "chosen_score": best_score,
            "final_epoch_score": final_epoch_score}


def summarize_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    runs = manifest.get("runs")
    horizon = manifest.get("expected_horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("manifest 'expected_horizon' must be a positive int")
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest 'runs' must be a nonempty list")
    seen = set()
    for e in runs:
        if isinstance(e, dict):
            key = (e.get("arm"), e.get("seed"))
            if key in seen:
                raise ValueError(f"duplicate manifest entry for arm {key[0]!r} seed {key[1]!r}")
            seen.add(key)
    results = [summarize_run(e, horizon) for e in runs]
    by_seed = {}
    for r in results:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r
    pairs = []
    for seed in sorted(by_seed):
        arms = by_seed[seed]
        g = arms.get("gold_only")
        p = arms.get("public4k")
        if g and p and g["status"] == "complete" and p["status"] == "complete":
            pairs.append({"seed": seed,
                          "gold_only": {"chosen_step": g["chosen_step"],
                                        "chosen_score": g["chosen_score"],
                                        "final_epoch_score": g["final_epoch_score"],
                                        "source_sha256": g["source_sha256"]},
                          "public4k": {"chosen_step": p["chosen_step"],
                                       "chosen_score": p["chosen_score"],
                                       "final_epoch_score": p["final_epoch_score"],
                                       "source_sha256": p["source_sha256"]},
                          "paired_difference": p["chosen_score"] - g["chosen_score"],
                          "paired_final_epoch_difference":
                              p["final_epoch_score"] - g["final_epoch_score"]})
    out = {"scope": "Aggregate receipt from retrieved train.jsonl logs; validation macro-F1 only",
           "expected_horizon": horizon,
           "n_runs": len(results),
           "n_complete": sum(1 for r in results if r["status"] == "complete"),
           "n_pairs": len(pairs),
           "seed_results": sorted(results, key=lambda r: (r["seed"], r["arm"])),
           "pairs": pairs,
           "uncertainty": "Descriptive only; no confidence intervals computed (small n)."}
    if len(pairs) >= 2:
        mean = sum(p["paired_difference"] for p in pairs) / len(pairs)
        mean_final = sum(p["paired_final_epoch_difference"] for p in pairs) / len(pairs)
        out["descriptive_mean_paired_difference"] = mean
        out["descriptive_mean_paired_final_epoch_difference"] = mean_final
        out["descriptive_note"] = ("Means of paired differences (public4k minus gold_only) "
                                   "for chosen and final-epoch scores; "
                                   "descriptive only, no CI.")
    else:
        out["descriptive_note"] = "Mean paired difference omitted: fewer than two complete pairs."
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt = summarize_manifest(manifest)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
