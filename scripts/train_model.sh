#!/bin/bash

# Define paths
PROJECT_DIR="/path/to/your/project"
NOTEBOOK="$PROJECT_DIR/modeling.ipynb"
OUTPUT_NOTEBOOK="$PROJECT_DIR/logs/modeling_output.ipynb"
LOG_FILE="$PROJECT_DIR/logs/train.log"

# Activate virtual environment (if using one)
source $PROJECT_DIR/venv/bin/activate  # Adjust if necessary

# Run the Jupyter Notebook using papermill
echo "🚀 Running Jupyter Notebook training..."
papermill "$NOTEBOOK" "$OUTPUT_NOTEBOOK" > "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed. Check logs: $LOG_FILE"
fi
