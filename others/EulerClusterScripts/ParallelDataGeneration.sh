#!/bin/bash
#SBATCH --array=1-2             # Create an array job with task IDs from 1 to 12
#SBATCH --cpus-per-task=1        # Assign the required number of CPUs per task
#SBATCH --time=8:00:00           # Set the maximum job time

source $HOME/miniconda3/bin/activate
conda activate CPS39

# Use SLURM_ARRAY_TASK_ID for the model index directly since it ranges from 1 to 50
i=$SLURM_ARRAY_TASK_ID

cd ../../
# Run the Python script with the specific index
python run_data_generator.py -i $i