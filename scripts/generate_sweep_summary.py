#!/usr/bin/env python3
"""
Generate a sweep-level sweep_summary.txt aggregating all run summaries.

Usage:
    python scripts/generate_sweep_summary.py --sweep-dir /path/to/sweep
"""

import argparse
import json
import re
import sys
from pathlib import Path


def is_gibberish(resp: str) -> bool:
    """Detect gibberish output."""
    if not resp:
        return False
    words = resp.split()
    return len(words) > 10 and len(set(words)) / len(words) < 0.15


def load_eval_json(path: Path) -> dict:
    """Load an eval JSON file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def count_gibberish(paired_path: Path) -> tuple:
    """Count gibberish in paired outputs."""
    if not paired_path.exists():
        return 0, 0
    pairs = []
    with open(paired_path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    gib = sum(1 for p in pairs if is_gibberish(p.get("cb_response", "")))
    return gib, len(pairs)


def _lm_tag_to_name(tag: str) -> str:
    """Convert loss mode short tag to full name."""
    return {"tf": "triplet_full", "cs": "cosine_simple"}.get(tag, tag)


def parse_run_name(name: str) -> dict:
    """Parse a run directory name into components."""
    result = {"preset": "default", "alpha": "?", "gamma_kl": "0.9",
              "loss_mode": "triplet_full", "layers": "?", "policy": "?"}

    # With loss mode tag + preset: preset_a10.0_g5.0_tf_l10_20_policy
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9]*)_a([\d.]+)_g([\d.]+)_([a-z]{2})_l([\d_]+)_(.+)$", name)
    if m:
        result["preset"] = m.group(1)
        result["alpha"] = m.group(2)
        result["gamma_kl"] = m.group(3)
        result["loss_mode"] = _lm_tag_to_name(m.group(4))
        result["layers"] = m.group(5).replace("_", ",")
        result["policy"] = m.group(6)
        return result

    # With loss mode tag, no preset: a10.0_g5.0_tf_l10_20_policy
    m = re.match(r"^a([\d.]+)_g([\d.]+)_([a-z]{2})_l([\d_]+)_(.+)$", name)
    if m:
        result["alpha"] = m.group(1)
        result["gamma_kl"] = m.group(2)
        result["loss_mode"] = _lm_tag_to_name(m.group(3))
        result["layers"] = m.group(4).replace("_", ",")
        result["policy"] = m.group(5)
        return result

    # With gamma, no loss mode tag: preset_a10.0_g5.0_l10_20_policy
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9]*)_a([\d.]+)_g([\d.]+)_l([\d_]+)_(.+)$", name)
    if m:
        result["preset"] = m.group(1)
        result["alpha"] = m.group(2)
        result["gamma_kl"] = m.group(3)
        result["layers"] = m.group(4).replace("_", ",")
        result["policy"] = m.group(5)
        return result

    # No preset, with gamma: a10.0_g5.0_l10_20_policy
    m = re.match(r"^a([\d.]+)_g([\d.]+)_l([\d_]+)_(.+)$", name)
    if m:
        result["alpha"] = m.group(1)
        result["gamma_kl"] = m.group(2)
        result["layers"] = m.group(3).replace("_", ",")
        result["policy"] = m.group(4)
        return result

    # Legacy with preset: preset_a10.0_l10_20_policy
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9]*)_a([\d.]+)_l([\d_]+)_(.+)$", name)
    if m:
        result["preset"] = m.group(1)
        result["alpha"] = m.group(2)
        result["layers"] = m.group(3).replace("_", ",")
        result["policy"] = m.group(4)
        return result

    # Legacy without preset: a10.0_l10_20_policy
    m = re.match(r"^a([\d.]+)_l([\d_]+)_(.+)$", name)
    if m:
        result["alpha"] = m.group(1)
        result["layers"] = m.group(2).replace("_", ",")
        result["policy"] = m.group(3)
        return result

    return result


def collect_run_metrics(run_dir: Path) -> dict:
    """Collect key metrics for a single run."""
    info = parse_run_name(run_dir.name)
    info["run_name"] = run_dir.name
    info["status"] = "unknown"

    # Try to load run_config.json for accurate config
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        info["preset"] = config.get("preset", info["preset"])
        info["alpha"] = str(config.get("alpha", info["alpha"]))
        info["gamma_kl"] = str(config.get("triplet_gamma_kl", info["gamma_kl"]))
        info["loss_mode"] = config.get("loss_mode", info.get("loss_mode", "triplet_full"))
        info["layers"] = config.get("layers", info["layers"])
        info["policy"] = config.get("policy", info["policy"])

    eval_dir = run_dir / "eval"

    # Fujitsu
    fujitsu = load_eval_json(eval_dir / "fujitsu_eval.json")
    gib_f, total_f = count_gibberish(eval_dir / "fujitsu_eval.paired_outputs.jsonl")

    if fujitsu:
        info["status"] = "success"
        b = fujitsu.get("baseline", {}).get("tool_flip_asr", {})
        c = fujitsu.get("cb_model", {}).get("tool_flip_asr", {})
        info["fuj_baseline_asr"] = b.get("attack_success_rate", "?")
        info["fuj_cb_asr"] = c.get("attack_success_rate", "?")
        info["fuj_correct"] = c.get("correct_tool_call_rate", c.get("correct_behavior_rate", "?"))
        info["fuj_no_tool"] = c.get("no_tool_call_rate", "?")
        info["fuj_gib"] = f"{gib_f}/{total_f}"
        delta = fujitsu.get("delta", {}).get("tool_flip_asr", "?")
        info["fuj_delta"] = delta

    # AgentDojo
    agentdojo = load_eval_json(eval_dir / "agentdojo_eval.json")
    gib_a, total_a = count_gibberish(eval_dir / "agentdojo_eval.paired_outputs.jsonl")

    if agentdojo:
        gc = agentdojo.get("cb_model", {}).get("generation_comparison", {})
        info["ad_resist"] = gc.get("harmful_resistance_rate", "?")
        info["ad_malicious"] = gc.get("malicious_tool_call_rate", "?")
        info["ad_gib"] = f"{gib_a}/{total_a}"

    # AgentDojo benign
    benign = load_eval_json(eval_dir / "agentdojo_benign_eval.json")
    if benign:
        gc = benign.get("cb_model", {}).get("generation_comparison", {})
        info["ad_benign_correct"] = gc.get("correct_tool_call_rate", "?")
        info["ad_benign_no_tool"] = gc.get("no_tool_call_rate", "?")

    return info


def fmt_pct(val) -> str:
    """Format a value as percentage string."""
    if isinstance(val, (int, float)):
        return f"{val:.1%}"
    return str(val) if val != "?" else "-"


def generate_sweep_summary(sweep_dir: Path) -> str:
    """Generate the full sweep summary."""
    parts = []
    parts.append("=" * 80)
    parts.append("SWEEP SUMMARY")
    parts.append("=" * 80)
    parts.append(f"Directory: {sweep_dir}")
    parts.append("")

    # Collect all run directories
    run_dirs = sorted([
        d for d in sweep_dir.iterdir()
        if d.is_dir() and (d / "run_config.json").exists()
    ])

    if not run_dirs:
        # Also try directories without run_config.json but with eval/
        run_dirs = sorted([
            d for d in sweep_dir.iterdir()
            if d.is_dir() and (d / "eval").exists()
        ])

    if not run_dirs:
        parts.append("No run directories found.")
        return "\n".join(parts)

    parts.append(f"Total runs: {len(run_dirs)}")
    parts.append("")

    # Collect metrics for all runs
    all_metrics = []
    for rd in run_dirs:
        metrics = collect_run_metrics(rd)
        all_metrics.append(metrics)

    # Results overview table
    parts.append("-" * 80)
    parts.append("RESULTS OVERVIEW")
    parts.append("-" * 80)
    parts.append("")

    # Table header
    header = (
        f"{'Run':<45} | {'Fuj ASR':>8} | {'Correct':>8} | {'Gib':>7} | "
        f"{'AD Resist':>9} | {'Benign':>8} | {'Status':>7}"
    )
    parts.append(header)
    parts.append("-" * len(header))

    for m in all_metrics:
        name = m["run_name"]
        if len(name) > 44:
            name = name[:41] + "..."

        fuj_asr = fmt_pct(m.get("fuj_cb_asr", "?"))
        fuj_correct = fmt_pct(m.get("fuj_correct", "?"))
        fuj_gib = m.get("fuj_gib", "-")
        ad_resist = fmt_pct(m.get("ad_resist", "?"))
        ad_benign = fmt_pct(m.get("ad_benign_correct", "?"))
        status = m.get("status", "?")

        parts.append(
            f"{name:<45} | {fuj_asr:>8} | {fuj_correct:>8} | {fuj_gib:>7} | "
            f"{ad_resist:>9} | {ad_benign:>8} | {status:>7}"
        )

    parts.append("")

    # Best configuration
    parts.append("-" * 80)
    parts.append("BEST CONFIGURATIONS (lowest CB ASR, no gibberish)")
    parts.append("-" * 80)
    parts.append("")

    valid = []
    for m in all_metrics:
        try:
            cb_asr = float(m.get("fuj_cb_asr", 999))
            gib_str = m.get("fuj_gib", "0/0")
            gib_n = int(gib_str.split("/")[0]) if "/" in gib_str else 0
            valid.append((cb_asr, gib_n, m))
        except (ValueError, TypeError):
            pass

    if valid:
        valid.sort(key=lambda x: (x[1] > 0, x[0]))
        for rank, (_asr, _gib, m) in enumerate(valid[:5], 1):
            parts.append(
                f"#{rank} {m['run_name']}"
            )
            parts.append(
                f"   Config: preset={m['preset']} alpha={m['alpha']} "
                f"gamma_kl={m['gamma_kl']} loss_mode={m.get('loss_mode', '?')} "
                f"layers={m['layers']} policy={m['policy']}"
            )
            parts.append(
                f"   Fujitsu: ASR={fmt_pct(m.get('fuj_cb_asr'))} "
                f"correct={fmt_pct(m.get('fuj_correct'))} "
                f"no_tool={fmt_pct(m.get('fuj_no_tool'))} "
                f"gibberish={m.get('fuj_gib', '-')}"
            )
            parts.append(
                f"   AgentDojo: resist={fmt_pct(m.get('ad_resist'))} "
                f"malicious={fmt_pct(m.get('ad_malicious'))} "
                f"benign_correct={fmt_pct(m.get('ad_benign_correct'))}"
            )
            parts.append("")
    else:
        parts.append("  (no valid results to rank)")
        parts.append("")

    # Concatenate individual run summaries
    parts.append("=" * 80)
    parts.append("INDIVIDUAL RUN SUMMARIES")
    parts.append("=" * 80)

    for rd in run_dirs:
        summary_path = rd / "run_summary.txt"
        if summary_path.exists():
            parts.append("")
            with open(summary_path) as f:
                parts.append(f.read())
        else:
            parts.append("")
            parts.append(f"--- {rd.name}: (no run_summary.txt) ---")

    parts.append("")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate sweep-level summary")
    parser.add_argument("--sweep-dir", required=True, help="Path to sweep directory")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"ERROR: Sweep directory not found: {sweep_dir}", file=sys.stderr)
        sys.exit(1)

    summary = generate_sweep_summary(sweep_dir)

    output_path = sweep_dir / "sweep_summary.txt"
    with open(output_path, "w") as f:
        f.write(summary)

    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
