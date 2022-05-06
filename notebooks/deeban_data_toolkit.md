# xSPI Data Toolkit
*Author: Deeban Ramalingam (2020)*

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

### Step 2. Mix datasets

We use [LatentSpaceBuilder](https://github.com/compSPI/LatentSpaceBuilder) to mix previously generated datasets.

| type | script | input |
| ---- | ------ | ----- |
| mixed | [cspi_create_synthetic_dataset_mixed_hit.py](https://github.com/compSPI/LatentSpaceBuilder/blob/master/latent_space_builder/cspi_create_synthetic_dataset_mixed_hit.py) | [cspi-create-synthetic-dataset-mixed-hit.json](https://github.com/compSPI/LatentSpaceBuilder/blob/master/latent_space_builder/cspi-create-synthetic-dataset-mixed-hit.json) |

To generate a mixed dataset, follow the instructions in the script itself. For example:
```bash
python cspi_create_synthetic_dataset_mixed_hit.py --config cspi-create-synthetic-dataset-mixed-hit.json --dataset 3iyf-10K-mixed-hit-99
```

### Step 3. Downsample mixed dataset

We use [LatentSpaceBuilder](https://github.com/compSPI/LatentSpaceBuilder) to downsample (featurize!) a dataset.

| type | script | input |
| ---- | ------ | ----- |
| downsampled | [incremental_pca_downsampling_featurization_mpi.py](https://github.com/compSPI/LatentSpaceBuilder/blob/master/latent_space_builder/incremental_pca_downsampling_featurization_mpi.py) | [incremental-pca-downsampling-featurization-mpi.json](https://github.com/compSPI/LatentSpaceBuilder/blob/master/latent_space_builder/incremental-pca-downsampling-featurization-mpi.json)

To downsample a dataset, follow the instructions in the script itself. For example:
```bash
mpiexec -n 16 python incremental_pca_downsampling_featurization_mpi.py --config incremental-pca-downsampling-featurization-mpi.json --dataset 3iyf-10K-mixed-hit-99-single-hits-labeled
```

### Step 4. Perform incremental PCA and label dataset

We use [LatentSpaceBuilder](https://github.com/compSPI/LatentSpaceBuilder) to perform incremental PCA and flag the dataset.

| type | script | input |
| ---- | ------ | ----- |
| IPCA | [flag_downsampled_diffraction_patterns_with_incremental_pca.py](https://github.com/compSPI/LatentSpaceBuilder/blob/master/latent_space_builder/flag_downsampled_diffraction_patterns_with_incremental_pca.py) | [flag-downsampled-diffraction-patterns-with-incremental-pca.json](https://github.com/compSPI/LatentSpaceBuilder/blob/master/latent_space_builder/flag-downsampled-diffraction-patterns-with-incremental-pca.json) |

To perform IPCA and flag a dataset, follow the instructions in the script itself. For example:
```bash
python flag_downsampled_diffraction_patterns_with_incremental_pca.py --config flag-downsampled-diffraction-patterns-with-incremental-pca.json --dataset 3iyf-10K-mixed-hit-99-single-hits-labeled
```

### Step 5. Visualize labelled merged dataset

We use [LatentSpaceVisualizer](https://github.com/compSPI/LatentSpaceVisualizer) to visualize a labelled or unlabelled dataset.

Follow the instructions in the [Jupyter notebook](https://github.com/compSPI/LatentSpaceVisualizer/blob/master/latent_space_visualizer.ipynb).
