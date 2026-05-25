#!/bin/bash
# Quick status check for EMNLP experiments

echo "=========================================="
echo "EMNLP EXPERIMENT STATUS"
echo "=========================================="

echo ""
echo "=== QUEUED JOBS ==="
squeue -u labiosa --format="%.10i %.9P %.30j %.2t %.10M %R" 2>/dev/null | head -20

echo ""
echo "=== ALREADY-PLANNED QWEN EXPERIMENTS ==="
echo "Exp 1 (Seed variance +1): Jobs 8677740_0-2"
echo "Exp 2 (Lambda=0):         Jobs 8677741_0-2"
echo "Exp 3 (Vanilla held-out): Job 8677742"

echo ""
echo "=== SCALE VARIATION EXPERIMENTS ==="
echo "Qwen2.5-1.5B LRD:   Job 8679939"
echo "Qwen2.5-14B LRD:    Job 8679940"
echo "Qwen2.5-14B Patch:  Job 8679941"
echo "Llama-3.2-1B LRD:   Job 8679942"

echo ""
echo "=== RESULTS DIRECTORIES ==="
ls -la /home1/labiosa/NLPSpring2026/Qwen2.5/scale_experiments/ 2>/dev/null || echo "Scale experiments dir not yet populated"
ls -la /home1/labiosa/NLPSpring2026/Llama/lrd_results_3.2_1B/ 2>/dev/null || echo "Llama 1B results not yet available"

echo ""
echo "=== CHECK SPECIFIC EXPERIMENT ==="
echo "Run: tail -f /home1/labiosa/NLPSpring2026/logs/<jobname>_<jobid>.out"
