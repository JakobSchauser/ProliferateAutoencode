#!/bin/bash
#
#SBATCH --job-name=NCA_run         # Job name
#SBATCH --output=logs/%x_%j.out        # Output log (%x=job name, %j=job ID) - change as needed
#SBATCH --error=logs/%x_%j.err         # Error log - change as needed
#SBATCH --partition=main               # Partition to submit to
#SBATCH --gres=gpu:1                   # Request GPU - Comment out or delete for no GPU
#SBATCH --cpus-per-task=1           # CPU cores per task
#SBATCH --mem=64G                      # RAM for job
#SBATCH --time=04:00:00                # Time limit hh:mm:ss 
#SBATCH --mail-type=END,FAIL           # Email notifications
#SBATCH --mail-user=youremail@domain   # Where to send email

echo "=== Job started on $(date) ==="
echo "Running on node: $(hostname)"

# ---- Load environment ----
ENV="NCA"  # <-- change this to your env name if needed
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate $ENV

# ---- Choose script to run ----
# Set either a Python file OR Notebook file below:
PYTHON_SCRIPT="run_coarse.py"        # e.g. scripts/train.py
NOTEBOOK_FILE=""        # e.g. notebooks/experiment.ipynb

# ---- Run a Python script (.py) ----
if [[ ! -z "$PYTHON_SCRIPT" ]]; then
    echo "Running Python script: $PYTHON_SCRIPT"
    python "$PYTHON_SCRIPT"
fi

# ---- Run a Jupyter notebook (.ipynb) ----
if [[ ! -z "$NOTEBOOK_FILE" ]]; then
    echo "Running Jupyter notebook: $NOTEBOOK_FILE"
    jupyter nbconvert --to python "$NOTEBOOK_FILE" --output temp_notebook.py
    python temp_notebook.py

    # Always clean up the temporary file
    echo "Cleaning up temporary file: temp_notebook.py"
    rm -f "temp_notebook.py"
fi

echo "=== Job finished on $(date) ==="