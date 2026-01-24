# Quick Fixes for Circuit Breaker Training

## Issue #1: Sparse Layer Targeting (2 vs 6 layers)

**Current**: Only using layers 10 and 20
**Fix**: Add more middle layers for better gradient signal

### Fix in Training Script

```bash
# In slurm/05_train.sbatch line 97
# BEFORE:
--cb-target-layers 10 20 \

# AFTER (recommended):
--cb-target-layers 10 12 14 16 18 20 \

# OR (minimal change):
--cb-target-layers 10 15 20 \
```

---

## Issue #2: Verify Completion Masking

**Problem**: Need to ensure loss is applied to tool call tokens, not prompt

### Check Existing Training Logs

```bash
# Look for validation output in recent runs
grep -A 30 "VALIDATING COMPLETION MASKING" slurm-*.out

# You should see:
#   - Mask covers <|python_tag|>: >90%
#   - Fallback to full sequence: <10%
```

### If Validation Shows Low Coverage (<50%)

The tokenization offset is wrong. Fix in `trainer.py:852`:

```python
# BEFORE (line 852):
prompt_tokens = self.tokenizer.encode(
    prompt_text, add_special_tokens=True
)

# AFTER:
prompt_tokens = self.tokenizer.encode(
    prompt_text, add_special_tokens=False  # Changed to False
)
```

**Why**: The full sequence already has special tokens. Adding them again during
prompt tokenization creates an offset mismatch.

---

## Issue #3: Add Diagnostic Logging

**Problem**: Can't tell if rerouting is working without monitoring cos_sim

### Add to `trainer.py` after line 1514

```python
# In _compute_step_loss, after line 1514 (where metrics are logged)
if self.global_step % 50 == 0:  # Every 50 steps
    self.accelerator.print(
        f"\n[Step {self.global_step}] "
        f"cs={cs:.3f} cr={cr:.3f} | "
        f"cos_sim={reroute_metrics['cos_sim_mean']:.4f} | "
        f"L_rr={loss_reroute.item():.4f} L_ret={loss_retain.item():.4f}"
    )
```

**Success Indicators**:
- `cos_sim` should **decrease** from ~0.8 toward 0.0
- If it stays high (>0.5) after 300+ steps, rerouting isn't working
- If `L_ret` explodes (>100), you're destroying capabilities

---

## Issue #4: Check Alpha Schedule is Active

### Verify Dual Coefficients Are Being Used

Add this ONE-TIME check at training start:

```python
# In trainer.py, in __init__ or first training step
print("\n=== ALPHA SCHEDULE CHECK ===")
for step in [0, 100, 250, 400, 500]:
    prog = min(step / (self.config.alpha_decay_multiplier * self.config.total_steps), 1.0)
    cs = self.config.alpha_max * (1.0 - prog)
    cr = self.config.alpha_max * prog
    print(f"Step {step:3d}: cs={cs:5.2f}  cr={cr:5.2f}")
print("="*30 + "\n")
```

**Expected output** (for `alpha_max=10`, `decay_multiplier=2`, `total_steps=500`):
```
Step   0: cs=10.00  cr= 0.00
Step 100: cs= 9.00  cr= 1.00
Step 250: cs= 7.50  cr= 2.50
Step 400: cs= 6.00  cr= 4.00
Step 500: cs= 5.00  cr= 5.00
```

If all values are the same, dual coefficients aren't working!

---

## Issue #5: Training Data Quantity

### Check Sample Counts

```bash
# Count harmful samples
wc -l data/processed/*harmful*.jsonl

# Count benign samples
wc -l data/processed/*benign*.jsonl
```

**Recommendation**:
- Minimum: 1000 harmful + 2000 benign
- Optimal: 5000 harmful + 10000 benign (like original paper)

If you have <1000 total, generate more data first.

---

## Priority Action Checklist

Run these in order:

### [ ] 1. Check existing validation output
```bash
# Find most recent SLURM output
ls -lt slurm-*.out | head -1

# Check masking validation
grep -A 30 "VALIDATING COMPLETION MASKING" <most_recent_slurm_file>
```

**Decision point**: If mask coverage <50%, fix tokenization offset (Issue #2)

---

### [ ] 2. Update training script with more layers
```bash
# Edit slurm/05_train.sbatch line 97
vim slurm/05_train.sbatch
# Change: --cb-target-layers 10 20
# To:     --cb-target-layers 10 12 14 16 18 20
```

---

### [ ] 3. Add diagnostic logging
```python
# Edit src/training/trainer.py
# Add the logging code from Issue #3 above
```

---

### [ ] 4. Run short test (50-100 steps)
```bash
# Submit training with reduced steps to verify fixes
sbatch slurm/05_train.sbatch
# OR modify script to set --total-steps 100
```

**Watch for**:
- Validation output shows >90% python_tag coverage ✓
- cos_sim starts high (~0.7-0.9) and begins decreasing ✓
- Alpha schedule shows cs decreasing, cr increasing ✓

---

### [ ] 5. Full training run
Once diagnostics look good, run full 500 steps

---

## Quick Test Script

Save this as `test_masking.py` and run before training:

```python
#!/usr/bin/env python3
"""
Test completion masking on a sample.
"""
import torch
from transformers import AutoTokenizer
import sys

# Load your tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Sample text (replace with actual from your data)
sample_text = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Can you search for cats?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|python_tag|>{"name": "search_web", "parameters": {"query": "cats"}}<|eom_id|>"""

# Tokenize full text
full_ids = tokenizer.encode(sample_text, add_special_tokens=False)
print(f"Full sequence length: {len(full_ids)}")

# Find assistant header end
import re
match = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>", sample_text)
if not match:
    print("ERROR: No assistant header found!")
    sys.exit(1)

completion_start_char = match.end()
print(f"Assistant starts at char: {completion_start_char}")

# Tokenize prompt (METHOD 1: with special tokens)
prompt_text = sample_text[:completion_start_char]
prompt_ids_with = tokenizer.encode(prompt_text, add_special_tokens=True)
print(f"Prompt length (add_special_tokens=True): {len(prompt_ids_with)}")

# Tokenize prompt (METHOD 2: without special tokens)
prompt_ids_without = tokenizer.encode(prompt_text, add_special_tokens=False)
print(f"Prompt length (add_special_tokens=False): {len(prompt_ids_without)}")

# Find <|python_tag|> position
python_tag_id = tokenizer.convert_tokens_to_ids("<|python_tag|>")
full_ids_tensor = torch.tensor(full_ids)
python_tag_positions = (full_ids_tensor == python_tag_id).nonzero(as_tuple=True)[0]

if len(python_tag_positions) > 0:
    python_tag_pos = python_tag_positions[0].item()
    print(f"\n<|python_tag|> at position: {python_tag_pos}")

    # Check which method captures it
    if python_tag_pos >= len(prompt_ids_with):
        print("✓ METHOD 1 (add_special_tokens=True) would MASK the tool call")
    else:
        print("✗ METHOD 1 would SKIP the tool call (BAD!)")

    if python_tag_pos >= len(prompt_ids_without):
        print("✓ METHOD 2 (add_special_tokens=False) would MASK the tool call")
    else:
        print("✗ METHOD 2 would SKIP the tool call (BAD!)")
else:
    print("\nWARNING: <|python_tag|> not found in sequence!")
```

Run it:
```bash
python test_masking.py
```

**Expected output**:
```
Full sequence length: 87
Assistant starts at char: 145
Prompt length (add_special_tokens=True): 65
Prompt length (add_special_tokens=False): 63

<|python_tag|> at position: 64
✓ METHOD 2 (add_special_tokens=False) would MASK the tool call
✗ METHOD 1 would SKIP the tool call (BAD!)
```

If METHOD 1 is correct in your case, keep `add_special_tokens=True`.
If METHOD 2 is correct, change to `False`.

---

## Expected Training Output (Success)

After fixes, you should see:

```
==================================================
VALIDATING COMPLETION MASKING
==================================================
  Checked 12 harmful samples from 3 batches:
  - Assistant header found: 12/12 (100.0%)
  - Mask covers <|python_tag|>: 12/12 (100.0%)
  - Fallback to full sequence: 0/12 (0.0%)

Training Configuration
==================================================
  Model: meta-llama/Llama-3.1-8B-Instruct
  CB Target Layers: [10, 12, 14, 16, 18, 20]
  Loss Weighting: dual
  Alpha Max: 10.0

[Step 0] cs=10.000 cr=0.000 | cos_sim=0.8234 | L_rr=0.6543 L_ret=2.1234
[Step 50] cs=9.500 cr=0.500 | cos_sim=0.7012 | L_rr=0.5234 L_ret=2.3456
[Step 100] cs=9.000 cr=1.000 | cos_sim=0.5678 | L_rr=0.4123 L_ret=2.4567
[Step 200] cs=8.000 cr=2.000 | cos_sim=0.3456 | L_rr=0.2345 L_ret=2.5678
[Step 400] cs=6.000 cr=4.000 | cos_sim=0.1234 | L_rr=0.0876 L_ret=2.6789
[Step 500] cs=5.000 cr=5.000 | cos_sim=0.0543 | L_rr=0.0321 L_ret=2.7123
```

**Key success indicators**:
- ✅ Mask covers 100% of python_tag tokens
- ✅ cos_sim decreases from ~0.8 to <0.1
- ✅ cs decreases, cr increases (dual schedule working)
- ✅ L_ret stays bounded (<10)

---

## If Still Not Working

After applying all fixes, if `cos_sim` still doesn't decrease:

### Hypothesis: Conceptual Mismatch

Your task (tool injection) may need a different approach than the original (harmful content):

1. **Focus loss on tool name tokens only**:
   ```python
   # Only mask the "search_web" or "retrieve_multimodal_docs" tokens
   # Not the entire completion
   ```

2. **Use adversarial contrastive loss**:
   ```python
   # Compare harmful (injected) vs benign (clean) for SAME query
   # Push them apart in representation space
   ```

3. **Check if injection patterns are in hidden states**:
   ```python
   # Print hidden state norms at injection position vs tool call position
   # If they're identical, model isn't "seeing" the injection
   ```

This would require architectural changes beyond quick fixes.
