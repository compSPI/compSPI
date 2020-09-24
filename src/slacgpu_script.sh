#!/bin/bash
#SBATCH --account=shared --partition=shared
#SBATCH --job-name=test
#SBATCH --output=output-%j.txt --error=output-%j.txt #
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1g
#SBATCH --time=10:00
#SBATCH --gpus=1
pwd; hostname; date

source ~/.bashrc

conda env list

conda activate compspi-vaegan

conda env list

python -c "import torch; print('CUDA availability: {}'.format(torch.cuda.is_available()))"
