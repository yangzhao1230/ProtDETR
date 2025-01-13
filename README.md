# ProtDETR

ProtDETR is a novel framework designed for the classification of multifunctional enzymes inspired by Object Detection in Computer Vision. You can read the preprint [here](https://arxiv.org/abs/2501.05644) - *Interpretable Enzyme Function Prediction via Residue-Level Detection*.

## Environment Requirements

To run ProtDETR, you will need to set up an environment with specific package versions. 

The required packages include but are not limited to:

- fair-esm-2.0.0
- torch-1.13.0 (GPU version, compatible with specific CUDA versions)
- scipy-1.10.1

## Data Requirements

### Multifunctional Enzyme Datasets

To facilitate the classification of multifunctional enzymes, please download the datasets (namely split100, new, and price) from [CLEAN's GitHub repository](https://github.com/tttianhao/CLEAN). Subsequently, move the downloaded CSV files into the `./data/multi_func/` directory within your project.

### Monofunctional Enzyme Dataset

For the classification of monofunctional enzymes, acquire the ECPred40 dataset available on Zenodo at <https://doi.org/10.5281/zenodo.7253910>. Ensure to convert the training, validation, and test sets into CSV format and position them in the `./data/mono_func/` directory of your project.

Alternatively, you may opt to download the pre-organized dataset directly from our Google Drive at [this link](https://drive.google.com/drive/folders/1g87b982Rt5kX46wpi7-zyWgGBEE_e9IN?usp=sharing). Simply place the entire `data` directory within the root of your project for seamless integration.

## Running Commands

### Training 
To train models for enzyme classification, use the following commands:

- **For multifunctional enzyme classification:**

  ```
  python train_multi.py
  ```

- **For monofunctional enzyme classification:**

  ```
  python train_mono.py
  ```

The default parameters utilized in our scripts are consistent with those detailed in our final publication. Users are encouraged to modify the hyperparameters to tailor them to their specific needs. To facilitate a more efficient training process, it is highly recommended to use Distributed Data Parallel (DDP) via `torchrun`. For practical implementations of `torchrun`, please consult the `train_multi.sh` and `train_mono.sh` scripts.

Ensure that the `saved_models` directory has already been created within your project directory to prevent any issues with model weight saving.

### Evaluating

- **For evaluating multifunctional enzyme classification models:**

  ```
  python eval_multi.py
  ```

- **For evaluating monofunctional enzyme classification models:**

  ```
  python eval_mono.py
  ```


### Inferring Important Sites (e.g., Active Sites)

This script allows for the input of a UniProt ID and identifies important sites, such as active sites, based on attention maps generated by our model.

**Run command:**
```bash
python infer_important_site.py
```

**Usage Example:**
When prompted, enter the UniProt ID and the active sites separated by commas as shown below:

```plaintext
Enter UniProt ID ('exit' to quit): A0JNI4
Enter active sites separated by commas (e.g., 32,47,140): 56, 84
```

**Results:**
The results are saved for the specified UniProt ID in the following format:

```plaintext
Results saved for A0JNI4 in ./uniprot_results/A0JNI4.txt
```

**Output File Content Example:**

- **Encoder Top N attention sites:**
  - 85, 91, 92, 90, 314, 169, 205, 232, 282, 193

- **Encoder detected active sites:**

- **Decoder Predictions and Attention Analysis:**

  - **Query 1:**
    - Predicted EC: 4.3.1.17
    - Decoder Top N attention sites: 56, 90, 85, 91, 92, 86, 84, 50, 55, 83
    - Detected active sites: 56, 84

## Pretrained Models

Pretrained models for ProtDETR are available on Google Drive at [this link](https://drive.google.com/drive/folders/1g87b982Rt5kX46wpi7-zyWgGBEE_e9IN?usp=sharing). Please download and place the `saved_models` directory within the root directory of your project.


## Acknowledgments

Our code is primarily based on the DETR framework by Facebook Research ([DETR GitHub repository](https://github.com/facebookresearch/detr)), the CLEAN method ([CLEAN GitHub repository](https://github.com/tttianhao/CLEAN)), and we have also incorporated code from the EnzBert project ([EnzBert GitLab repository](https://gitlab.inria.fr/nbuton/tfpc)).

## Citation

@misc{yang2025interpretableenzymefunctionprediction,  
      title={Interpretable Enzyme Function Prediction via Residue-Level Detection},  
      author={Zhao Yang and Bing Su and Jiahao Chen and Ji-Rong Wen},  
      year={2025},  
      eprint={2501.05644},  
      archivePrefix={arXiv},  
      primaryClass={q-bio.BM},  
      url={https://arxiv.org/abs/2501.05644}  
}

