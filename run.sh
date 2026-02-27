#!/bin/bash

# ════════════════════════════════════════════════════════════════════════════
# Step 1: Pre-training (single run, fixed seed=1337)
# ════════════════════════════════════════════════════════════════════════════

echo "========================================"
echo "Pre-training (seed=1)"
echo "========================================"
python toyGPT_pretrain.py

if [ $? -ne 0 ]; then
    echo "Pre-training failed. Exiting."
    exit 1
fi

# ════════════════════════════════════════════════════════════════════════════
# Step 2: Fine-tuning (multiple seeds)
# ════════════════════════════════════════════════════════════════════════════

for seed in 1 1337 222 42 0; do
    echo "========================================"
    echo "Fine-tuning seed=$seed"
    echo "========================================"
    python toyGPT_addition.py --seed $seed
done
