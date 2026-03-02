#!/usr/bin/env python3
"""
Generate a human-readable run_summary.txt for a single training run.

Usage:
    python scripts/generate_run_summary.py --run-dir /path/to/run
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_run_config(run_dir: Path) -> dict:
    """Load run_config.json."""
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def format_config_section(config: dict) -> str:
    """Format configuration as readable text."""
    lines = []
    lines.append("1. CONFIGURATION")
    lines.append("-" * 60)

    pairs = [
        ("Preset", config.get("preset", "?")),
        ("Alpha", config.get("alpha", "?")),
        ("Gamma KL", config.get("triplet_gamma_kl", "?")),
        ("Layers", config.get("layers", "?")),
        ("Policy", config.get("policy", "?")),
        ("Pooling Mode", config.get("pooling_mode", "?")),
        ("Distance", config.get("distance", "?")),
        ("Margins", f"{config.get('margin_benign', '?')} / {config.get('margin_harmful', '?')}"),
        ("Total Steps", config.get("total_steps", "?")),
        ("Batch / Accum", f"{config.get('batch_size', '?')} / {config.get('grad_accum', '?')}"),
        ("Learning Rate", config.get("learning_rate", "?")),
        ("Loss Mode", config.get("loss_mode", "?")),
        ("Loss Weighting", config.get("loss_weighting", "?")),
        ("Pooling Policy", config.get("pooling_policy", "") or "(none)"),
        ("Pooling Mask", config.get("pooling_mask_policy", "?")),
        ("LoRA r/alpha", f"{config.get('lora_r', '?')} / {config.get('lora_alpha', '?')}"),
    ]

    # Two-column layout
    mid = (len(pairs) + 1) // 2
    col1 = pairs[:mid]
    col2 = pairs[mid:]
    for i in range(max(len(col1), len(col2))):
        left = ""
        right = ""
        if i < len(col1):
            k, v = col1[i]
            left = f"  {k + ':':<18}{v}"
        if i < len(col2):
            k, v = col2[i]
            right = f"  {k + ':':<18}{v}"
        lines.append(f"{left:<40}{right}")

    return "\n".join(lines)


def parse_train_log(run_dir: Path) -> dict:
    """Parse train.log for diagnostics and training history."""
    log_path = run_dir / "train.log"
    result = {
        "decoupled_masks": None,
        "pool_mask_info": [],
        "pooling_diagnostic_block": [],
        "steps": [],
        "raw_diagnostic": [],
    }

    if not log_path.exists():
        return result

    with open(log_path) as f:
        lines = f.readlines()

    in_diagnostic = False
    diagnostic_lines = []

    for line in lines:
        stripped = line.rstrip()

        # Decoupled mask status
        m = re.search(r"Decoupled masks: (ACTIVE|INACTIVE)", stripped)
        if m:
            result["decoupled_masks"] = m.group(1)

        # Pool mask active counts
        m = re.search(r"(Harmful|Benign) pool mask active:\s*(\d+) vs label mask active: (\d+)", stripped)
        if m:
            result["pool_mask_info"].append({
                "type": m.group(1),
                "pool_active": int(m.group(2)),
                "label_active": int(m.group(3)),
            })

        # Pooling diagnostic block
        if "[POOLING DIAGNOSTIC]" in stripped:
            in_diagnostic = True
            diagnostic_lines = [stripped]
            continue
        if in_diagnostic:
            if stripped.strip() == "" and len(diagnostic_lines) > 3:
                in_diagnostic = False
                result["pooling_diagnostic_block"] = diagnostic_lines
            else:
                diagnostic_lines.append(stripped)

        # Step metrics
        m = re.match(
            r"Step (\d+): mode=(\w+), loss=([\d.]+)"
            r"(?:.*?triplet_b=([\d.]+))?"
            r"(?:.*?triplet_h=([\d.]+))?"
            r"(?:.*?triplet_kl=([\d.]+))?"
            r"(?:.*?kl=([\d.]+))?"
            r"(?:.*?(?:α|alpha)=([\d.]+))?",
            stripped,
        )
        if m:
            step_data = {
                "step": int(m.group(1)),
                "mode": m.group(2),
                "loss": float(m.group(3)),
                "triplet_b": float(m.group(4)) if m.group(4) else None,
                "triplet_h": float(m.group(5)) if m.group(5) else None,
                "triplet_kl": float(m.group(6)) if m.group(6) else None,
                "kl": float(m.group(7)) if m.group(7) else None,
                "alpha": float(m.group(8)) if m.group(8) else None,
            }
            result["steps"].append(step_data)

    # If diagnostic block was still open at EOF
    if in_diagnostic and diagnostic_lines:
        result["pooling_diagnostic_block"] = diagnostic_lines

    return result


def format_diagnostic_section(log_data: dict) -> str:
    """Format pooling diagnostic section."""
    lines = []
    lines.append("")
    lines.append("2. POOLING DIAGNOSTIC")
    lines.append("-" * 60)

    status = log_data.get("decoupled_masks")
    if status:
        lines.append(f"  Decoupled masks: {status}")
    else:
        lines.append("  Decoupled masks: (not found in log)")

    for info in log_data.get("pool_mask_info", []):
        lines.append(
            f"  {info['type']} pool mask active: {info['pool_active']} "
            f"vs label mask active: {info['label_active']}"
        )

    diag_block = log_data.get("pooling_diagnostic_block", [])
    if diag_block:
        lines.append("")
        for dl in diag_block[:30]:  # Limit to 30 lines
            lines.append(f"  {dl.rstrip()}")

    if not status and not diag_block:
        lines.append("  (no diagnostic output found in train.log)")

    return "\n".join(lines)


def format_training_history(log_data: dict) -> str:
    """Format training history as a table."""
    lines = []
    lines.append("")
    lines.append("3. TRAINING HISTORY")
    lines.append("-" * 60)

    steps = log_data.get("steps", [])
    if not steps:
        lines.append("  (no training steps found in train.log)")
        return "\n".join(lines)

    # Determine which columns are available
    has_triplet = any(s.get("triplet_b") is not None for s in steps)
    if has_triplet:
        header = f"  {'Step':>5} | {'Loss':>10} | {'Triplet_B':>10} | {'Triplet_H':>10} | {'KL':>10} | {'Alpha':>8}"
        sep = f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}"
        lines.append(header)
        lines.append(sep)
        for s in steps:
            tb = f"{s['triplet_b']:.4f}" if s.get("triplet_b") is not None else "-"
            th = f"{s['triplet_h']:.4f}" if s.get("triplet_h") is not None else "-"
            kl = s.get("triplet_kl") or s.get("kl")
            kl_str = f"{kl:.4f}" if kl is not None else "-"
            alpha = f"{s['alpha']:.4f}" if s.get("alpha") is not None else "-"
            lines.append(
                f"  {s['step']:>5} | {s['loss']:>10.4f} | {tb:>10} | {th:>10} | {kl_str:>10} | {alpha:>8}"
            )
    else:
        header = f"  {'Step':>5} | {'Loss':>10} | {'Alpha':>8}"
        sep = f"  {'-'*5}-+-{'-'*10}-+-{'-'*8}"
        lines.append(header)
        lines.append(sep)
        for s in steps:
            alpha = f"{s['alpha']:.4f}" if s.get("alpha") is not None else "-"
            lines.append(f"  {s['step']:>5} | {s['loss']:>10.4f} | {alpha:>8}")

    # Summary statistics
    losses = [s["loss"] for s in steps]
    lines.append("")
    lines.append(f"  First loss: {losses[0]:.4f}")
    lines.append(f"  Last loss:  {losses[-1]:.4f}")
    lines.append(f"  Min loss:   {min(losses):.4f}")
    lines.append(f"  Max loss:   {max(losses):.4f}")

    # Detect triplet collapse (if triplet_h collapses to 0)
    if has_triplet:
        triplet_h_vals = [(s["step"], s["triplet_h"]) for s in steps if s.get("triplet_h") is not None]
        collapse_steps = [step for step, val in triplet_h_vals if val == 0.0]
        if collapse_steps:
            lines.append(f"  Triplet_H collapse: first zero at step {collapse_steps[0]}")
        else:
            lines.append(f"  Triplet_H collapse: none (good)")

    return "\n".join(lines)


def is_gibberish(resp: str) -> bool:
    """Detect gibberish output (repetitive text)."""
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
    """Count gibberish in paired outputs. Returns (gib_count, total_count)."""
    if not paired_path.exists():
        return 0, 0
    pairs = []
    with open(paired_path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    gib = sum(1 for p in pairs if is_gibberish(p.get("cb_response", "")))
    return gib, len(pairs)


def format_eval_section(run_dir: Path) -> str:
    """Format evaluation metrics."""
    lines = []
    lines.append("")
    lines.append("4. EVALUATION METRICS")
    lines.append("-" * 60)

    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        lines.append("  (no eval directory found)")
        return "\n".join(lines)

    # Fujitsu
    fujitsu = load_eval_json(eval_dir / "fujitsu_eval.json")
    fujitsu_paired = eval_dir / "fujitsu_eval.paired_outputs.jsonl"
    gib_f, total_f = count_gibberish(fujitsu_paired)

    if fujitsu:
        b = fujitsu.get("baseline", {}).get("tool_flip_asr", {})
        c = fujitsu.get("cb_model", {}).get("tool_flip_asr", {})
        delta = fujitsu.get("delta", {}).get("tool_flip_asr", "?")

        lines.append("  FUJITSU:")
        if b:
            lines.append(f"    Baseline ASR:     {b.get('attack_success_rate', '?'):.1%}" if isinstance(b.get('attack_success_rate'), (int, float)) else f"    Baseline ASR:     {b.get('attack_success_rate', '?')}")
        if c:
            asr = c.get("attack_success_rate", "?")
            correct = c.get("correct_tool_call_rate", c.get("correct_behavior_rate", "?"))
            no_tool = c.get("no_tool_call_rate", "?")
            other = c.get("other_tool_call_rate", "?")
            malformed = c.get("malformed_tool_call_rate", "?")
            lines.append(f"    CB ASR:           {asr:.1%}" if isinstance(asr, (int, float)) else f"    CB ASR:           {asr}")
            lines.append(f"    CB Correct:       {correct:.1%}" if isinstance(correct, (int, float)) else f"    CB Correct:       {correct}")
            lines.append(f"    CB No Tool:       {no_tool:.1%}" if isinstance(no_tool, (int, float)) else f"    CB No Tool:       {no_tool}")
            lines.append(f"    CB Other Tool:    {other:.1%}" if isinstance(other, (int, float)) else f"    CB Other Tool:    {other}")
            lines.append(f"    CB Malformed:     {malformed:.1%}" if isinstance(malformed, (int, float)) else f"    CB Malformed:     {malformed}")
        if isinstance(delta, (int, float)):
            lines.append(f"    Delta (ASR):      {delta:+.1%}")
        lines.append(f"    Gibberish:        {gib_f}/{total_f}")
    else:
        lines.append("  FUJITSU: (no eval data)")

    # AgentDojo harmful
    agentdojo = load_eval_json(eval_dir / "agentdojo_eval.json")
    agentdojo_paired = eval_dir / "agentdojo_eval.paired_outputs.jsonl"
    gib_a, total_a = count_gibberish(agentdojo_paired)

    if agentdojo:
        lines.append("")
        lines.append("  AGENTDOJO (harmful):")
        oc = agentdojo.get("output_comparison", {})
        if oc:
            lines.append(f"    Diff Rate:        {oc.get('difference_rate', '?'):.1%}" if isinstance(oc.get("difference_rate"), (int, float)) else f"    Diff Rate:        {oc.get('difference_rate', '?')}")
        for key in ("baseline", "cb_model"):
            gc = agentdojo.get(key, {}).get("generation_comparison", {})
            if gc:
                label = "Baseline" if key == "baseline" else "CB Model"
                mal = gc.get("malicious_tool_call_rate", "?")
                res = gc.get("harmful_resistance_rate", "?")
                cor = gc.get("correct_tool_call_rate", "?")
                no = gc.get("no_tool_call_rate", "?")
                parts = []
                if isinstance(mal, (int, float)):
                    parts.append(f"malicious={mal:.0%}")
                if isinstance(res, (int, float)):
                    parts.append(f"resist={res:.0%}")
                if isinstance(cor, (int, float)):
                    parts.append(f"correct={cor:.0%}")
                if isinstance(no, (int, float)):
                    parts.append(f"no_tool={no:.0%}")
                lines.append(f"    {label}: {' '.join(parts)}")
        lines.append(f"    Gibberish:        {gib_a}/{total_a}")
    else:
        lines.append("\n  AGENTDOJO (harmful): (no eval data)")

    # AgentDojo benign
    benign = load_eval_json(eval_dir / "agentdojo_benign_eval.json")
    benign_paired = eval_dir / "agentdojo_benign_eval.paired_outputs.jsonl"
    gib_b, total_b = count_gibberish(benign_paired)

    if benign:
        lines.append("")
        lines.append("  AGENTDOJO (benign):")
        for key in ("baseline", "cb_model"):
            gc = benign.get(key, {}).get("generation_comparison", {})
            if gc:
                label = "Baseline" if key == "baseline" else "CB Model"
                cor = gc.get("correct_tool_call_rate", "?")
                no = gc.get("no_tool_call_rate", "?")
                parts = []
                if isinstance(cor, (int, float)):
                    parts.append(f"correct_tool={cor:.0%}")
                if isinstance(no, (int, float)):
                    parts.append(f"no_tool={no:.0%}")
                lines.append(f"    {label}: {' '.join(parts)}")
        lines.append(f"    Gibberish:        {gib_b}/{total_b}")
    else:
        lines.append("\n  AGENTDOJO (benign): (no eval data)")

    # Checkpoint progression
    ckpt_dir = eval_dir / "checkpoints"
    if ckpt_dir.exists():
        def _ckpt_sort_key(name: str) -> int:
            m = re.search(r"(\d+)", name)
            return int(m.group(1)) if m else 0

        ckpt_names = sorted(
            [d.name for d in ckpt_dir.iterdir() if d.is_dir()],
            key=_ckpt_sort_key,
        )
        if ckpt_names:
            lines.append("")
            lines.append("  CHECKPOINT PROGRESSION:")
            for ckpt_name in ckpt_names:
                ckpt_eval = load_eval_json(ckpt_dir / ckpt_name / "fujitsu_eval.json")
                if ckpt_eval:
                    c = ckpt_eval.get("cb_model", {}).get("tool_flip_asr", {})
                    asr = c.get("attack_success_rate", "?")
                    correct = c.get("correct_tool_call_rate", c.get("correct_behavior_rate", "?"))
                    no_tool = c.get("no_tool_call_rate", "?")
                    asr_str = f"{asr:.1%}" if isinstance(asr, (int, float)) else str(asr)
                    cor_str = f"{correct:.1%}" if isinstance(correct, (int, float)) else str(correct)
                    no_str = f"{no_tool:.1%}" if isinstance(no_tool, (int, float)) else str(no_tool)
                    lines.append(f"    {ckpt_name}: ASR={asr_str}  correct={cor_str}  no_tool={no_str}")

    return "\n".join(lines)


def format_examples_section(run_dir: Path, max_examples: int = 5) -> str:
    """Format baseline vs CB example comparisons."""
    lines = []
    lines.append("")
    lines.append("5. EXAMPLE COMPARISONS")
    lines.append("-" * 60)

    eval_dir = run_dir / "eval"
    shown = 0

    # Fujitsu examples
    fujitsu_paired = eval_dir / "fujitsu_eval.paired_outputs.jsonl"
    if fujitsu_paired.exists():
        with open(fujitsu_paired) as f:
            pairs = [json.loads(line) for line in f if line.strip()]

        if pairs:
            improvements = [
                p for p in pairs
                if p.get("baseline_outcome") == "attack_success"
                and p.get("cb_outcome") != "attack_success"
            ]
            regressions = [
                p for p in pairs
                if p.get("baseline_outcome") != "attack_success"
                and p.get("cb_outcome") == "attack_success"
            ]
            selected = (improvements + regressions)[:max_examples]

            if selected:
                lines.append("  FUJITSU:")
                for i, p in enumerate(selected, 1):
                    b_out = p.get("baseline_outcome", "?")
                    c_out = p.get("cb_outcome", "?")
                    label = "IMP" if b_out == "attack_success" else "REG"
                    gib = " [GIBBERISH]" if is_gibberish(p.get("cb_response", "")) else ""
                    lines.append(f"  [{i}] {label}: {b_out} -> {c_out}{gib}")
                    lines.append(
                        f"      expected={p.get('expected_tool', '?')} "
                        f"simulated={p.get('simulated_tool', '?')}"
                    )
                    lines.append("  --- BASELINE ---")
                    lines.append(f"  {(p.get('baseline_response') or '(empty)')[:300]}")
                    lines.append("  --- CB MODEL ---")
                    lines.append(f"  {(p.get('cb_response') or '(empty)')[:300]}")
                    lines.append("")
                shown += len(selected)

    # AgentDojo examples (if room)
    remaining = max_examples - shown
    if remaining > 0:
        agentdojo_paired = eval_dir / "agentdojo_eval.paired_outputs.jsonl"
        if agentdojo_paired.exists():
            with open(agentdojo_paired) as f:
                pairs = [json.loads(line) for line in f if line.strip()]

            differ = [p for p in pairs if p.get("responses_differ")]
            selected = differ[:remaining]

            if selected:
                lines.append("  AGENTDOJO:")
                for i, p in enumerate(selected, 1):
                    cat = p.get("category", "?")
                    b_tool = p.get("baseline_observed_tool") or "none"
                    c_tool = p.get("cb_observed_tool") or "none"
                    gib = " [GIBBERISH]" if is_gibberish(p.get("cb_response", "")) else ""
                    lines.append(f"  [{i}] {cat}: baseline={b_tool} -> cb={c_tool}{gib}")
                    lines.append("  --- BASELINE ---")
                    lines.append(f"  {(p.get('baseline_response') or '(empty)')[:300]}")
                    lines.append("  --- CB MODEL ---")
                    lines.append(f"  {(p.get('cb_response') or '(empty)')[:300]}")
                    lines.append("")

    if shown == 0 and not (eval_dir / "agentdojo_eval.paired_outputs.jsonl").exists():
        lines.append("  (no paired output files found)")

    return "\n".join(lines)


def generate_summary(run_dir: Path) -> str:
    """Generate the full summary text."""
    config = parse_run_config(run_dir)
    log_data = parse_train_log(run_dir)

    run_name = config.get("run_name", run_dir.name)

    parts = []
    parts.append("=" * 60)
    parts.append(f"RUN SUMMARY: {run_name}")
    parts.append("=" * 60)
    parts.append(f"Directory: {run_dir}")
    parts.append(f"Timestamp: {config.get('timestamp', '?')}")
    parts.append(f"Job ID:    {config.get('job_id', '?')}")
    parts.append("")
    parts.append(format_config_section(config))
    parts.append(format_diagnostic_section(log_data))
    parts.append(format_training_history(log_data))
    parts.append(format_eval_section(run_dir))
    parts.append(format_examples_section(run_dir))
    parts.append("")
    parts.append("=" * 60)
    parts.append(f"END OF SUMMARY: {run_name}")
    parts.append("=" * 60)

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate per-run summary text file")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    summary = generate_summary(run_dir)

    output_path = run_dir / "run_summary.txt"
    with open(output_path, "w") as f:
        f.write(summary)

    print(f"Generated: {output_path}")

    # Also print to stdout for convenience
    print(summary)


if __name__ == "__main__":
    main()
