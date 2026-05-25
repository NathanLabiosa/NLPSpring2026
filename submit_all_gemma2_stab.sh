#!/bin/bash
# Submit all Gemma2 training jobs with L_stab in parallel

echo "Submitting 7 Gemma2 training jobs with stability loss..."

for script in Gemma2/submit_train_L*_stab.slurm; do
    jobid=$(sbatch $script | awk '{print $NF}')
    echo "Submitted $(basename $script): job $jobid"
done

echo ""
echo "All 7 Gemma2 jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
