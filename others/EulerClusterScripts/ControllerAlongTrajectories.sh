#!/bin/bash
#SBATCH --array=1-192             
#SBATCH --cpus-per-task=1        # Assign the required number of CPUs per task
#SBATCH --mem-per-cpu=1G         
#SBATCH --time=8:00:00           # Set the maximum job time
#SBATCH --output=./others/EulerClusterScripts/EulerTerminalOutput/Experiment_16_11_2024_pole_L_and_m_random/slurm-%A_%a.out   # Output file

# Create output and error directories if they do not exist
mkdir -p ./others/EulerClusterScripts/EulerTerminalOutput/Experiment_16_11_2024_pole_L_and_m_random

source $HOME/miniconda3/bin/activate
conda activate CPS39

# Use SLURM_ARRAY_TASK_ID for the model index directly since it ranges from 1 to 50
i=$SLURM_ARRAY_TASK_ID

# Run the Python script with the specific index
python ./SI_Toolkit_ASF/Run/PreprocessData_Add_Control_Along_Trajectories.py -i $i

