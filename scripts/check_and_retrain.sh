#!/bin/bash

python src/monitoring/compute_drift.py

AVG_PSI=$(python -c "import json; print(json.load(open('drift_report.json'))['avg_psi'])")

echo "Average PSI: $AVG_PSI"

THRESHOLD=0.2

if (( $(echo "$AVG_PSI > $THRESHOLD" | bc -l) )); then
    echo "Drift detected. Retraining..."
    dvc repro
else
    echo "No significant drift."
fi
