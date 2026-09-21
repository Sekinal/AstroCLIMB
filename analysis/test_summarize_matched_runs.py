"""Tests for summarize_matched_runs using synthetic actual-schema logs."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

MOD_PATH = Path(__file__).with_name("summarize_matched_runs.py")
_SPEC = importlib.util.spec_from_file_location("summarize_matched_runs", MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["summarize_matched_runs"] = _mod
_SPEC.loader.exec_module(_mod)

summarize_manifest = _mod.summarize_manifest
summarize_run = _mod.summarize_run

H = 837
STEPS = [279, 558, 837]
TRAIN_ROWS = 2230
VAL_ROWS = 794
SENTINEL = object()


def write_log(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def ev(event, **kw):
    return json.dumps({"event": event, **kw})


def init_for_arm(arm):
    return None if arm == "gold_only" else "public-4k-init"


def complete_log(path, scores, done_step=H, guard=True, val_every=279, seed=7,
                 arm="gold_only", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                 init_checkpoint=SENTINEL, done_event="done"):
    if init_checkpoint is SENTINEL:
        init_checkpoint = init_for_arm(arm)
    cfg = {"seed": seed, "val_every": val_every, "init_checkpoint": init_checkpoint}
    lines = [ev("ready", train_rows=train_rows, val_rows=val_rows,
                total_steps=H, config=cfg)]
    for step, macro in zip(STEPS, scores):
        lines.append(ev("checkpoint", step=step, reload_verified=guard))
        lines.append(ev("validation", step=step, macro=macro, best=macro))
    lines.append(ev(done_event, step=done_step, best=max(scores)))
    return write_log(path, lines)


def ready_of(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("event") == "ready":
            return obj
    raise AssertionError("no ready event in log")


def test_tiebreak_earliest_and_pairing(tmp_path):
    g = complete_log(tmp_path / "g7.jsonl", [0.5, 0.6, 0.6], seed=7, arm="gold_only")
    p = complete_log(tmp_path / "p7.jsonl", [0.55, 0.62, 0.61], seed=7, arm="public4k")
    m = {"expected_horizon": H, "runs": [
        {"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0},
        {"arm": "public4k", "seed": 7, "log": p, "exitcode": 0}]}
    out = summarize_manifest(m)
    assert out["n_pairs"] == 1
    assert "descriptive_mean_paired_difference" not in out
    gold = next(r for r in out["seed_results"] if r["arm"] == "gold_only")
    assert gold["chosen_step"] == 558 and gold["chosen_score"] == pytest.approx(0.6)
    assert gold["final_epoch_score"] == pytest.approx(0.6)
    assert out["pairs"][0]["paired_difference"] == pytest.approx(0.02)
    assert out["pairs"][0]["paired_final_epoch_difference"] == pytest.approx(0.01)


def test_final_epoch_score_uses_exact_horizon(tmp_path):
    g = complete_log(tmp_path / "g.jsonl", [0.9, 0.1, 0.2], seed=7, arm="gold_only")
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0}, H)
    assert r["status"] == "complete"
    assert r["chosen_step"] == 279 and r["chosen_score"] == pytest.approx(0.9)
    assert r["final_epoch_score"] == pytest.approx(0.2)


def test_ready_rowcounts_are_top_level_not_nested(tmp_path):
    g = complete_log(tmp_path / "g.jsonl", [0.5, 0.6, 0.61], seed=7, arm="gold_only")
    ready = ready_of(g)
    assert ready["train_rows"] == TRAIN_ROWS
    assert ready["val_rows"] == VAL_ROWS
    assert "train_rows" not in ready["config"]
    assert "val_rows" not in ready["config"]
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0}, H)
    assert r["status"] == "complete"


def test_top_level_rowcounts_authoritative_over_nested(tmp_path):
    path = tmp_path / "g.jsonl"
    cfg = {"seed": 7, "val_every": 279, "init_checkpoint": None,
           "train_rows": 1, "val_rows": 2}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    for step, macro in zip(STEPS, [0.5, 0.6, 0.61]):
        lines.append(ev("checkpoint", step=step, reload_verified=True))
        lines.append(ev("validation", step=step, macro=macro, best=macro))
    lines.append(ev("done", step=H, best=0.61))
    write_log(path, lines)
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": str(path), "exitcode": 0}, H)
    assert r["status"] == "complete"
    assert r["chosen_step"] == 837


def test_wrong_top_level_rejected_despite_correct_nested(tmp_path):
    path = tmp_path / "g.jsonl"
    cfg = {"seed": 7, "val_every": 279, "init_checkpoint": None,
           "train_rows": TRAIN_ROWS, "val_rows": VAL_ROWS}
    lines = [ev("ready", train_rows=1000, val_rows=100,
                total_steps=H, config=cfg)]
    for step, macro in zip(STEPS, [0.5, 0.6, 0.61]):
        lines.append(ev("checkpoint", step=step, reload_verified=True))
        lines.append(ev("validation", step=step, macro=macro, best=macro))
    lines.append(ev("done", step=H, best=0.61))
    write_log(path, lines)
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": str(path), "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r
    assert "ready.train_rows" in r.get("reason", "")


def test_incomplete_nonzero_exit_has_no_score(tmp_path):
    g = complete_log(tmp_path / "g.jsonl", [0.5, 0.6, 0.61], seed=7, arm="gold_only")
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": g, "exitcode": 3}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r and "chosen_step" not in r
    assert "final_epoch_score" not in r


def test_incomplete_wrong_final_step(tmp_path):
    g = complete_log(tmp_path / "g.jsonl", [0.5, 0.6, 0.61], done_step=558,
                     seed=7, arm="gold_only")
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r


def test_nan_score_raises(tmp_path):
    path = tmp_path / "g.jsonl"
    cfg = {"seed": 7, "val_every": 279, "init_checkpoint": None}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    for step, macro in [(279, 0.5), (558, float("nan")), (837, 0.6)]:
        lines.append(ev("checkpoint", step=step, reload_verified=True))
        raw = json.dumps({"event": "validation", "step": step, "macro": macro, "best": 0.5})
        lines.append(raw)
    lines.append(ev("done", step=H, best=0.6))
    write_log(path, lines)
    with pytest.raises(ValueError):
        summarize_run({"arm": "gold_only", "seed": 7, "log": str(path), "exitcode": 0}, H)


def test_wrong_validation_steps_raise(tmp_path):
    g = complete_log(tmp_path / "g.jsonl", [0.5, 0.6, 0.61], seed=7, arm="gold_only")
    objs = [json.loads(line) for line in (tmp_path / "g.jsonl").read_text().splitlines()]
    for o in objs:
        if o.get("event") == "validation" and o.get("step") == 279:
            o["step"] = 280
    (tmp_path / "g.jsonl").write_text("\n".join(json.dumps(o) for o in objs) + "\n")
    with pytest.raises(ValueError):
        summarize_run({"arm": "gold_only", "seed": 7, "log": str(tmp_path / "g.jsonl"), "exitcode": 0}, H)


def test_missing_arm_gives_no_pair_but_preserves_seeds(tmp_path):
    g = complete_log(tmp_path / "g7.jsonl", [0.5, 0.6, 0.61], seed=7, arm="gold_only")
    m = {"expected_horizon": H, "runs": [{"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0}]}
    out = summarize_manifest(m)
    assert out["pairs"] == [] and len(out["seed_results"]) == 1
    assert out["seed_results"][0]["status"] == "complete"


def test_pairing_differing_seeds_gives_no_pair(tmp_path):
    g = complete_log(tmp_path / "g7.jsonl", [0.5, 0.6, 0.61], seed=7, arm="gold_only")
    p = complete_log(tmp_path / "p19.jsonl", [0.5, 0.6, 0.62], seed=19, arm="public4k")
    m = {"expected_horizon": H, "runs": [
        {"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0},
        {"arm": "public4k", "seed": 19, "log": p, "exitcode": 0}]}
    out = summarize_manifest(m)
    assert out["pairs"] == []
    assert len(out["seed_results"]) == 2
    assert "descriptive_mean_paired_difference" not in out


def test_malformed_json_raises_without_private_path(tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    path = sub / "bad.jsonl"
    path.write_text(ev("ready", total_steps=H) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        summarize_run({"arm": "gold_only", "seed": 7, "log": str(path), "exitcode": 0}, H)
    msg = str(ei.value)
    assert "bad.jsonl" in msg
    assert str(path) not in msg
    assert str(tmp_path) not in msg


def test_duplicate_validation_raises(tmp_path):
    path = tmp_path / "g.jsonl"
    cfg = {"seed": 7, "val_every": 279, "init_checkpoint": None}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    for step, macro in [(279, 0.5), (279, 0.55), (837, 0.6)]:
        lines.append(ev("checkpoint", step=step, reload_verified=True))
        lines.append(ev("validation", step=step, macro=macro, best=macro))
    lines.append(ev("done", step=H, best=0.6))
    write_log(path, lines)
    with pytest.raises(ValueError):
        summarize_run({"arm": "gold_only", "seed": 7, "log": str(path), "exitcode": 0}, H)


def test_blank_lines_ignored(tmp_path):
    g = complete_log(tmp_path / "g.jsonl", [0.5, 0.6, 0.61], seed=7, arm="gold_only")
    with open(g, "a", encoding="utf-8") as f:
        f.write("\n   \n")
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0}, H)
    assert r["status"] == "complete" and r["chosen_step"] == 837


def test_reload_guard_fail_is_incomplete(tmp_path):
    g = complete_log(tmp_path / "g.jsonl", [0.5, 0.6, 0.61], guard=False,
                     seed=7, arm="gold_only")
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r


def test_no_checkpoints_is_incomplete(tmp_path):
    cfg = {"seed": 7, "val_every": 279, "init_checkpoint": None}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    for step, macro in zip(STEPS, [0.5, 0.6, 0.61]):
        lines.append(ev("validation", step=step, macro=macro, best=macro))
    lines.append(ev("done", step=H, best=0.61))
    path = write_log(tmp_path / "g.jsonl", lines)
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": path, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r


def test_missing_single_verified_checkpoint_is_incomplete(tmp_path):
    cfg = {"seed": 7, "val_every": 279, "init_checkpoint": None}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    for step, macro in zip(STEPS, [0.5, 0.6, 0.61]):
        if step != 558:
            lines.append(ev("checkpoint", step=step, reload_verified=True))
        lines.append(ev("validation", step=step, macro=macro, best=macro))
    lines.append(ev("done", step=H, best=0.61))
    path = write_log(tmp_path / "g.jsonl", lines)
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": path, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r


def test_mean_only_with_two_pairs_and_no_paths(tmp_path):
    g7 = complete_log(tmp_path / "g7.jsonl", [0.5, 0.6, 0.61], seed=7, arm="gold_only")
    p7 = complete_log(tmp_path / "p7.jsonl", [0.5, 0.6, 0.63], seed=7, arm="public4k")
    g19 = complete_log(tmp_path / "g19.jsonl", [0.5, 0.5, 0.5], seed=19, arm="gold_only")
    p19 = complete_log(tmp_path / "p19.jsonl", [0.5, 0.5, 0.55], seed=19, arm="public4k")
    m = {"expected_horizon": H, "runs": [
        {"arm": "gold_only", "seed": 7, "log": g7, "exitcode": 0},
        {"arm": "public4k", "seed": 7, "log": p7, "exitcode": 0},
        {"arm": "gold_only", "seed": 19, "log": g19, "exitcode": 0},
        {"arm": "public4k", "seed": 19, "log": p19, "exitcode": 0}]}
    out = summarize_manifest(m)
    assert out["n_pairs"] == 2
    assert out["descriptive_mean_paired_difference"] == pytest.approx(0.035)
    assert out["descriptive_mean_paired_final_epoch_difference"] == pytest.approx(0.035)
    assert out["pairs"][0]["paired_final_epoch_difference"] == pytest.approx(0.02)
    blob = json.dumps(out)
    assert "/tmp" not in blob and "private" not in blob
    for r in out["seed_results"]:
        assert "log" not in r
    assert math.isfinite(out["descriptive_mean_paired_difference"])
    assert math.isfinite(out["descriptive_mean_paired_final_epoch_difference"])


def test_duplicate_manifest_entry_raises(tmp_path):
    g = complete_log(tmp_path / "g7.jsonl", [0.5, 0.6, 0.61], seed=7, arm="gold_only")
    m = {"expected_horizon": H, "runs": [
        {"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0},
        {"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0}]}
    with pytest.raises(ValueError):
        summarize_manifest(m)


def test_config_seed_mismatch_is_incomplete(tmp_path):
    g = complete_log(tmp_path / "g.jsonl", [0.5, 0.6, 0.61], seed=19, arm="gold_only")
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r
    assert "7" in r.get("reason", "")


def test_train_val_rows_mismatch_is_incomplete(tmp_path):
    g = complete_log(tmp_path / "g.jsonl", [0.5, 0.6, 0.61], seed=7,
                     arm="gold_only", train_rows=1000)
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r
    assert "ready.train_rows" in r.get("reason", "")
    g2 = complete_log(tmp_path / "g2.jsonl", [0.5, 0.6, 0.61], seed=7,
                      arm="gold_only", val_rows=100)
    r2 = summarize_run({"arm": "gold_only", "seed": 7, "log": g2, "exitcode": 0}, H)
    assert r2["status"] == "incomplete"
    assert "chosen_score" not in r2
    assert "ready.val_rows" in r2.get("reason", "")
    for path in (g, g2):
        ready = ready_of(path)
        assert "train_rows" not in ready["config"]
        assert "val_rows" not in ready["config"]


def test_init_checkpoint_per_arm(tmp_path):
    bad_pub = complete_log(tmp_path / "p.jsonl", [0.5, 0.6, 0.61], seed=7,
                           arm="public4k", init_checkpoint=None)
    r = summarize_run({"arm": "public4k", "seed": 7, "log": bad_pub, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r
    bad_gold = complete_log(tmp_path / "g.jsonl", [0.5, 0.6, 0.61], seed=7,
                            arm="gold_only", init_checkpoint="something")
    r2 = summarize_run({"arm": "gold_only", "seed": 7, "log": bad_gold, "exitcode": 0}, H)
    assert r2["status"] == "incomplete"
    assert "chosen_score" not in r2
    good_pub = complete_log(tmp_path / "p2.jsonl", [0.5, 0.6, 0.61], seed=7, arm="public4k")
    r3 = summarize_run({"arm": "public4k", "seed": 7, "log": good_pub, "exitcode": 0}, H)
    assert r3["status"] == "complete"


def test_val_every_mismatch_and_missing_is_incomplete(tmp_path):
    g = complete_log(tmp_path / "g.jsonl", [0.5, 0.6, 0.61], seed=7,
                     arm="gold_only", val_every=100)
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": g, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r
    cfg = {"seed": 7, "init_checkpoint": None}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    for step, macro in zip(STEPS, [0.5, 0.6, 0.61]):
        lines.append(ev("checkpoint", step=step, reload_verified=True))
        lines.append(ev("validation", step=step, macro=macro, best=macro))
    lines.append(ev("done", step=H, best=0.61))
    path = write_log(tmp_path / "g2.jsonl", lines)
    r2 = summarize_run({"arm": "gold_only", "seed": 7, "log": path, "exitcode": 0}, H)
    assert r2["status"] == "incomplete"
    assert "chosen_score" not in r2


def test_alternate_schedule_not_accepted(tmp_path):
    cfg = {"seed": 7, "val_every": 100, "init_checkpoint": None}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    alt_steps = [100, 200, 300, 400, 500, 600, 700, 800, 837]
    for step in alt_steps:
        lines.append(ev("checkpoint", step=step, reload_verified=True))
        lines.append(ev("validation", step=step, macro=0.5, best=0.5))
    lines.append(ev("done", step=H, best=0.5))
    path = write_log(tmp_path / "g.jsonl", lines)
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": path, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r


def test_done_before_final_validation_is_incomplete(tmp_path):
    cfg = {"seed": 7, "val_every": 279, "init_checkpoint": None}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    lines.append(ev("checkpoint", step=279, reload_verified=True))
    lines.append(ev("validation", step=279, macro=0.5, best=0.5))
    lines.append(ev("done", step=H, best=0.5))
    for step, macro in [(558, 0.6), (837, 0.61)]:
        lines.append(ev("checkpoint", step=step, reload_verified=True))
        lines.append(ev("validation", step=step, macro=macro, best=macro))
    path = write_log(tmp_path / "g.jsonl", lines)
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": path, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r


def test_last_final_event_wins(tmp_path):
    cfg = {"seed": 7, "val_every": 279, "init_checkpoint": None}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    for step, macro in zip(STEPS, [0.5, 0.6, 0.61]):
        lines.append(ev("checkpoint", step=step, reload_verified=True))
        lines.append(ev("validation", step=step, macro=macro, best=macro))
    lines.append(ev("done", step=558, best=0.6))
    lines.append(ev("done", step=H, best=0.61))
    path = write_log(tmp_path / "g.jsonl", lines)
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": path, "exitcode": 0}, H)
    assert r["status"] == "complete"
    assert r["final_epoch_score"] == pytest.approx(0.61)


def test_partial_deadline_run_is_incomplete_without_score(tmp_path):
    cfg = {"seed": 7, "val_every": 279, "init_checkpoint": None}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    lines.append(ev("checkpoint", step=279, reload_verified=True))
    lines.append(ev("validation", step=279, macro=0.5, best=0.5))
    path = write_log(tmp_path / "g.jsonl", lines)
    r = summarize_run({"arm": "gold_only", "seed": 7, "log": path, "exitcode": 0}, H)
    assert r["status"] == "incomplete"
    assert "chosen_score" not in r and "chosen_step" not in r
    assert "final_epoch_score" not in r


def test_partial_run_bad_scores_still_raise(tmp_path):
    cfg = {"seed": 7, "val_every": 279, "init_checkpoint": None}
    lines = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                total_steps=H, config=cfg)]
    lines.append(ev("checkpoint", step=279, reload_verified=True))
    lines.append(ev("validation", step=279, macro=2.0, best=0.5))
    path = write_log(tmp_path / "bad.jsonl", lines)
    with pytest.raises(ValueError):
        summarize_run({"arm": "gold_only", "seed": 7, "log": path, "exitcode": 0}, H)
    lines2 = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                 total_steps=H, config=cfg)]
    lines2.append(ev("checkpoint", step=279, reload_verified=True))
    lines2.append(ev("validation", step=279, macro=0.5, best=0.5))
    lines2.append(ev("checkpoint", step=279, reload_verified=True))
    lines2.append(ev("validation", step=279, macro=0.6, best=0.6))
    path2 = write_log(tmp_path / "dup.jsonl", lines2)
    with pytest.raises(ValueError):
        summarize_run({"arm": "gold_only", "seed": 7, "log": path2, "exitcode": 0}, H)
    lines3 = [ev("ready", train_rows=TRAIN_ROWS, val_rows=VAL_ROWS,
                 total_steps=H, config=cfg)]
    lines3.append(ev("checkpoint", step=280, reload_verified=True))
    lines3.append(ev("validation", step=280, macro=0.5, best=0.5))
    path3 = write_log(tmp_path / "unexp.jsonl", lines3)
    with pytest.raises(ValueError):
        summarize_run({"arm": "gold_only", "seed": 7, "log": path3, "exitcode": 0}, H)
