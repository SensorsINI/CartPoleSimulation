#!/bin/bash
#SBATCH --array=1-240             # Create an array job with task IDs from 1 to 12
#SBATCH --cpus-per-task=1        # Assign the required number of CPUs per task
#SBATCH --time=8:00:00           # Set the maximum job time
#SBATCH --output=./others/EulerClusterScripts/slurm-%A_%a.out   # Output file

# Create output and error directories if they do not exist
mkdir -p ./others/EulerClusterScripts

source $HOME/miniconda3/bin/activate
conda activate CPS39

# Use SLURM_ARRAY_TASK_ID for the model index directly since it ranges from 1 to 50
i=$SLURM_ARRAY_TASK_ID

# Run the Python script with the specific index

python ./SI_Toolkit_ASF/Run/PreprocessData_Add_Control_Along_Trajectories_previous_up.py -i $i
python ./SI_Toolkit_ASF/Run/PreprocessData_Add_Control_Along_Trajectories_smooth_up.py -i $i