# ProtDETR

ProtDETR is a novel framework designed for the classification of multifunctional enzymes. This README provides detailed instructions on setting up the environment, preparing data, and running commands.

**Note:** We are currently organizing a clean and easy-to-use version of the codebase for ProtDETR. It is still being uploaded and will be completed shortly. Please check back soon.

## Environment Requirements

To run ProtDETR, you will need to set up an environment with specific package versions. We are in the process of finalizing a conda environment that will encapsulate all the necessary dependencies in a clean manner. This environment will be available for download soon.

The required packages include but are not limited to:

- fair-esm-2.0.0
- torch-1.13.0 (GPU version, compatible with specific CUDA versions)
- scipy-1.10.1

You can temporarily install these packages using pip. For the GPU version of torch, make sure to specify the version that matches your CUDA installation:

## Data Requirements

### Multifunctional Enzyme Datasets

For multifunctional enzyme classification, download the datasets (split100, new, price) from [CLEAN's GitHub repository](https://github.com/tttianhao/CLEAN). Place the corresponding CSV files into the `./data/multi_func/` directory.

### Monofunctional Enzyme Dataset

For monofunctional enzyme classification, download the ECPred40 dataset from Zenodo: <https://doi.org/10.5281/zenodo.7253910>. Convert the training, validation, and test sets to the same CSV format and place them in the `./data/mono_func/` directory.

## Running Commands

To train models for enzyme classification, use the following commands:

- For multifunctional enzyme classification:

      python train_multi.py

- For monofunctional enzyme classification:

      python train_mono.py

## Pretrained Models

Pretrained models for ProtDETR will be made available on Google Drive soon. Stay tuned for updates.

## Acknowledgments

Our code is primarily based on the DETR framework by Facebook Research ([DETR GitHub repository](https://github.com/facebookresearch/detr)), the CLEAN method ([CLEAN GitHub repository](https://github.com/tttianhao/CLEAN)), and we have also incorporated code from the EnzBert project ([EnzBert GitLab repository](https://gitlab.inria.fr/nbuton/tfpc)). We express our gratitude to the developers of these projects for their groundbreaking work that facilitated the development of ProtDETR.
