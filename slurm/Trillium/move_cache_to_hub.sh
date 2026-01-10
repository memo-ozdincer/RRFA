#!/bin/bash
# =============================================================================
# Move Cached Models from transformers/ to hub/ (One-Time Fix)
# =============================================================================
# Run this ONCE on the login node to fix the cache location mismatch
# This avoids re-downloading ~16GB+ of models
#
# Usage:
#   bash slurm/Trillium/move_cache_to_hub.sh
# =============================================================================

set -euo pipefail

CACHE_ROOT="/scratch/memoozd/cb_cache/hf"
OLD_CACHE="$CACHE_ROOT/transformers"
NEW_CACHE="$CACHE_ROOT/hub"

echo "========================================"
echo "Moving Models to Standard Hub Cache"
echo "========================================"
echo "From: $OLD_CACHE"
echo "To:   $NEW_CACHE"
echo ""

# Check source directory exists
if [[ ! -d "$OLD_CACHE" ]]; then
    echo "ERROR: Old cache not found at $OLD_CACHE"
    exit 1
fi

# Check if there are models to move
MODEL_COUNT=$(find "$OLD_CACHE" -maxdepth 1 -type d -name 'models--*' | wc -l)
if [[ $MODEL_COUNT -eq 0 ]]; then
    echo "No models found in $OLD_CACHE - nothing to move"
    exit 0
fi

echo "Found $MODEL_COUNT model(s) to move:"
find "$OLD_CACHE" -maxdepth 1 -type d -name 'models--*' -exec basename {} \;
echo ""

# Create target directory if needed
mkdir -p "$NEW_CACHE"

# Move each model
echo "Moving models..."
for MODEL_DIR in "$OLD_CACHE"/models--*; do
    if [[ -d "$MODEL_DIR" ]]; then
        MODEL_NAME=$(basename "$MODEL_DIR")
        echo "  Moving $MODEL_NAME..."
        
        # Check if target already exists
        if [[ -d "$NEW_CACHE/$MODEL_NAME" ]]; then
            echo "    ⚠️  Target already exists, skipping: $NEW_CACHE/$MODEL_NAME"
            continue
        fi
        
        # Move the model
        mv "$MODEL_DIR" "$NEW_CACHE/"
        echo "    ✅ Moved to $NEW_CACHE/$MODEL_NAME"
    fi
done

echo ""
echo "========================================"
echo "Verification"
echo "========================================"
echo "Hub cache contents:"
ls -1 "$NEW_CACHE"/models--* 2>/dev/null || echo "(none)"
echo ""

echo "Old transformers cache contents:"
ls -1 "$OLD_CACHE"/models--* 2>/dev/null || echo "(empty - all moved!)"
echo ""

# Check disk usage
echo "Disk usage:"
echo "  Hub cache: $(du -sh "$NEW_CACHE" 2>/dev/null | cut -f1)"
if [[ -d "$OLD_CACHE" ]]; then
    OLD_SIZE=$(du -sh "$OLD_CACHE" 2>/dev/null | cut -f1)
    echo "  Old transformers cache: $OLD_SIZE"
    if [[ "$OLD_SIZE" == "0"* || "$OLD_SIZE" == "4.0K"* ]]; then
        echo ""
        echo "Old cache is now empty. You can remove it:"
        echo "  rmdir $OLD_CACHE"
    fi
fi
echo ""

echo "========================================"
echo "✅ Cache migration complete!"
echo "========================================"
echo ""
echo "Your models are now in the standard Hub cache location."
echo "You can now run your SLURM jobs:"
echo "  sbatch slurm/Trillium/trillium_mvp_generate_ds.sbatch"
echo ""
