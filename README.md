## Pseudocell Tracer: Inferring cellular trajectories from scRNA-seq


### Prerequisites
* Pseudocell Tracer is tested to work on Python 3.7+
* Pseudocell Tracer requires the following Python libraries:
  - Tensorflow 1.14
  - Numpy 1.17.2
  - Pandas 0.25.1
  - Scipy 1.3.1
  - Scikit-learn 0.21.3
  - Matplotlib 3.0.3
  - Seaborn 0.9.0
  - Umap-learn 0.3.10

A Conda Python environment is provided in pseudocell_tracer.yml

```bash
conda env create -f pseudocell_tracer.yml
```

### Usage

```bash
pseudocell_tracer.py [-h] DATA SIDE_DATA OUTPUT_DIR [--plot_style PLOT_STYLE] [--num_cells NUM_CELLS] 
                          [--num_steps NUM_STEPS] --start START [START ...] --end END [END ...] 
                          [--genes GENES [GENES ...]]
```

```
Perform Pseudocell Tracer Algorithm

positional arguments:
  DATA                        Tab delimited file representing matrix of samples by genes
  SIDE_DATA                   Tab delimited file for side information to be used
  OUTPUT_DIR                  Output directory

optional arguments:
  -h, --help                  show this help message and exit
  --plot_style PLOT_STYLE     Use UMAP or tSNE for plotting (Default: UMAP)
  --num_cells NUM_CELLS       Number of pseudocells to generate at each step (Default: 100)
  --num_steps NUM_STEPS       Number of pseudocell states (Default: 100)
  --start START [START ...]   List of starting pseudocell states
  --end END [END ...]         List of ending pseudocell states
  --genes GENES [GENES ...]   Genes to plot in pseudocell trajectory
```

### Example for provided dataset

```bash
python pseudocell_tracer.py data/mnn.nocos.full.genes.tsv data/ighc.genes.relative.tsv output_dir --start Ighm --end Ighg1 Ighg2b Ighg3 --genes Aicda Bach2
```

The provided command will run Pseudocell Tracer on the provided scRNA-Seq data (provided in ZIP format) and store results in the directory _output_dir_. This run will infer three trajectories: IghM to IghG1, IghM to IghG2b, and IghM to IghG3. For each trajectory, 100 pseudocell states are generated for each step over 100 steps. The expression of the genes denoting starting and stopping states are plotted in addition to the additional genes specified: Aicda and Bach2. Neural network hyper-parameters can bet set in _config.py_.

The output directory will contain the following files:
|File | Description|
|---|---|
|run_network_config.txt | Copy of config.py used
|run_parameters.txt | Copy of command line parameters|
|encoder.h5 | Encoder model|
|decoder.h5 | Decoder model|
|input_scatter.png | Visual representation of input (Observed)|
|latent_scatter.png | Visual representation of latent space (Observed)|
|reconstructed_scatter.png | Visual representation of reconstruction (Observed)|
|generated_latent_scatter.png | Visual representation of latent space (Generated)|
|generated_latent_reconstruction.png | Visual representation of reconstruction (Generated)|

In addition there will be sub directories for inferred trajectories will contain the following:
|File | Description|
|---|---|
|generated_data.tsv | Tab separated file for generated gene expression data|
|generated_latent_data.tsv | Tab separated file for generated latent gene expression data|
|generated_side_data.tsv | Tab separated file for generated side data|
|genes.png | Line plot showing the trajectories of selected genes|

### Version
1.0.0 (2020/05/06)

### Publication
TBA

### Contact
* Please contact Aly Azeem Khan <aakhan@uchciago.edu> for any questions or comments.
* More recent updates to Pseudocell Tracer have been contributed by Derek Reiman <dreima2@uic.edu>

### License
Software provided to academic users under MIT License
