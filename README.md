# ProtDETR

ProtDETR is a novel framework designed for the classification of multifunctional enzymes. This README provides detailed instructions on setting up the environment, preparing data, and running training commands for both enzyme types.

## Environment Requirements

To run ProtDETR, you need to set up an environment with specific package versions. The required packages are:

- fair-esm-2.0.0
- torch-1.13.0

You can install these packages using pip:

    pip install fair-esm==2.0.0 torch==1.13.0

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

Our code is primarily based on the DETR framework by Facebook Research ([DETR GitHub repository](https://github.com/facebookresearch/detr)) and the CLEAN method ([CLEAN GitHub repository](https://github.com/tttianhao/CLEAN)). We express our gratitude to the developers of these projects for their groundbreaking work that facilitated the development of ProtDETR.