#!/bin/bash
# Submit all Qwen training jobs with L_stab in parallel

echo "Submitting 7 Qwen training jobs with stability loss..."

for script in Qwen2.5/submit_train_L*_stab.slurm; do
    jobid=$(sbatch $script | awk '{print $NF}')
    echo "Submitted $(basename $script): job $jobid"
done

echo ""
echo "All 7 Qwen jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
