#!/bin/bash
# Master script to submit all Qwen2.5-7B experiments
#
# Execution order:
# 1. Experiment 1: λ=0 ablation (3 seeds) - 12 hours
# 2. Experiment 2a: Held-out severity eval (uses existing checkpoints) - 2 hours
# 3. Experiment 2b-train: Retrain without speech - 8 hours
# 4. Experiment 2b-eval: Evaluate on held-out type - 2 hours (after 2b-train)
#
# Total estimated time: ~24 hours if run sequentially
# Can parallelize: 1 and 2a can run simultaneously; 2b-eval must wait for 2b-train

cd /home1/labiosa/NLPSpring2026/models/qwen2.5/experiments

echo "=========================================="
echo "Submitting Qwen2.5-7B Experiments"
echo "=========================================="

# Experiment 1: λ=0 ablation (array job with 3 seeds)
JOB1=$(sbatch --parsable submit_exp1_lambda0_ablation.slurm)
echo "Experiment 1 (λ=0 ablation, 3 seeds): Job $JOB1"

# Experiment 2a: Held-out severity (independent, uses existing checkpoints)
JOB2A=$(sbatch --parsable submit_exp2_heldout_severity.slurm)
echo "Experiment 2a (held-out severity): Job $JOB2A"

# Experiment 2b: Held-out type training (array job: vanilla + stabilizer)
JOB2B_TRAIN=$(sbatch --parsable submit_exp2_heldout_type_train.slurm)
echo "Experiment 2b-train (retrain w/o speech): Job $JOB2B_TRAIN"

# Experiment 2b: Held-out type eval (depends on training)
JOB2B_EVAL=$(sbatch --parsable --dependency=afterok:$JOB2B_TRAIN submit_exp2_heldout_type_eval.slurm)
echo "Experiment 2b-eval (eval on speech): Job $JOB2B_EVAL (waits for $JOB2B_TRAIN)"

echo ""
echo "=========================================="
echo "Jobs submitted:"
echo "  Exp 1 (λ=0):       $JOB1"
echo "  Exp 2a (severity): $JOB2A"
echo "  Exp 2b-train:      $JOB2B_TRAIN"
echo "  Exp 2b-eval:       $JOB2B_EVAL"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results will be in: ./results/"
echo "Run 'python aggregate_results.py' after jobs complete"
