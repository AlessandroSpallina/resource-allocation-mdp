#!/bin/bash
#
#SBATCH --job-name=slicing
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=all
#SBATCH --output=log.stdout
#SBATCH --error=log.stderr
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

date

source /mnt/beegfs/home/spallina/slicing5g/WORKSPACE/Slicing-5G-MDP/slicing-venv/bin/activate

which python

/mnt/beegfs/home/spallina/slicing5g/WORKSPACE/python/install/bin/python3 -m src.batch_manager.main --wdir /mnt/beegfs/home/spallina/slicing5g/WORKSPACE/Slicing-5G-MDP/src/batch_manager/


#/mnt/beegfs/home/spallina/slicing5g/WORKSPACE/python/install/bin/python3 -m src.slicing_core.main --wdir /mnt/beegfs/home/spallina/slicing5g/WORKSPACE/Slicing-5G-MDP/src/slicing_core/
