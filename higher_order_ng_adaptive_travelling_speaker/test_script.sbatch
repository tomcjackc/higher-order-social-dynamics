#!/bin/bash

#SBATCH --job-name Marius_Script
#SBATCH --time 3-00:00:00
#SBATCH --mail-type END
#SBATCH --mail-user marius.ignat19@imperial.ac.uk

python3 = '/usr/bin/python3'

$python3 ./test_traveling_speaker.py > out

#end


