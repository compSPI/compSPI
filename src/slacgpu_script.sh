#!/bin/bash
#SBATCH --account=shared --partition=shared
#SBATCH --job-name=test
#SBATCH --output=output-%j.txt --error=output-%j.txt #
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1g
#SBATCH --time=10:00
#SBATCH --gpus=1
echo "hello world"
# conda env create -f ../environment.yml
# python -c "print('hello world')"
