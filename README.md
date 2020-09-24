# compSPI
Computational Single Particle Imager.

## Installation

## xSPI Data Toolkit

### Step 1. Generate datasets

We use [pysingfel/Angstrom](https://github.com/chuckie82/pysingfel/tree/master/examples/scripts) to generate single-hits and double-hits datasets. At the moment, each of those uses its own script (will simplify later). Each script needs a .json input file.

| type | script | input |
| ---- | ------ | ----- |
| single-hit | [cspi_generate_synthetic_dataset_mpi_hybrid.py](https://github.com/chuckie82/pysingfel/blob/master/examples/scripts/cspi_generate_synthetic_dataset_mpi_hybrid.py) | [cspi_generate_synthetic_dataset_config.json](https://github.com/chuckie82/pysingfel/blob/master/examples/scripts/cspi_generate_synthetic_dataset_config.json) |
| double-hit | [cspi_generate_synthetic_dataset_double-hit_mpi_hybrid.py](https://github.com/chuckie82/pysingfel/blob/master/examples/scripts/cspi_generate_synthetic_dataset_double-hit_mpi_hybrid.py) | [cspi_generate_synthetic_dataset_config.json](https://github.com/chuckie82/pysingfel/blob/master/examples/scripts/cspi_generate_synthetic_dataset_config.json) |

To generate a given dataset, follow the instructions in the script itself. For example:
```bash
mpiexec -n 16 python cspi_generate_synthetic_dataset_mpi_hybrid.py --config cspi_generate_synthetic_dataset_config.json --dataset 3iyf-10K
```

### Step 2. Merge datasets

### Step 3. Downsample merged dataset

### Step 4. Perform incremental PCA

### Step 5. Visualize labelled merged dataset
