#!/bin/bash

#SBATCH --job-name TEST
#SBATCH --time 0-02:00:00
#SBATCH --mail-type END
#SBATCH --mail-user tjc119@imperial.ac.uk

# Create a local directory to run in.
scratch=/workspace/$USER/scratch/$SLURM_JOB_ID
mkdir -p $scratch

# Copy input file (called in this case input_file) to the directory job will
# run in. Slurm will start in the directory you submit your job from - so be
# sure this is in the home or data directory as workspace isn't shared between
# nodes.
cp higher-order-social-dynamics/ $scratch
cd $scratch

# Run program (test_prog.x in this example).
echo Executing in $scratch on $(hostname)
sh outcasts/steadystate.sh > output

# Copy back to submit directory. This is wrapped in an if statement so the
# working copy is only cleaned up if it has been copied back.
if cp output $SLURM_SUBMIT_DIR
then
  # Clean up.
  rm -rf $scratch
else
  echo "Copy failed. Data retained in $scratch on $(hostname)"
fi