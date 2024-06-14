

# Candidate Pseudolabel Learning


[ICML 2024 oral] This repository contains the official code for the ICML 2024 paper ["Candidate Pseudolabel Learning: Enhancing Vision-Language Models by Prompt Tuning with Unlabeled Data."](https://openreview.net/pdf?id=sBJNokmYuV) The research introduces a novel method, Candidate Pseudolabel Learning (CPL), which addresses the challenges of incorrect hard pseudolabels in fine-tuning vision-language models (VLMs) with unlabeled data. CPL refines the generation of candidate pseudolabels through both intra- and inter-instance label selection based on confidence score matrix, leading to improved label accuracy and class balance. 

<div align="center">
    <img src="imgs/overview.png" alt="overview" width="600">
</div>

## Table of Contents

- [Environment Setup](#environment-setup)
- [Reproducing the Main Results](#reproducing-the-main-results)
  - [Data Preparation](#data-preparation)
  - [Running the Experiments](#running-the-experiments)
- [Reproducing the Results of CPL+LaFTer](#reproducing-the-results-of-cpllafter)
  - [Data Preparation](#data-preparation-1)
  - [Running the Experiments](#running-the-experiments-1)
- [Citation](#citation)

## Environment Setup

In this project, we use Python 3.9.12 and the dependencies listed in the `requirements.txt` file. To install all the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Additionally, you need to manually install the `dassl` library in your environment by following the instructions [HERE](https://github.com/KaiyangZhou/Dassl.pytorch#installation).

## Reproducing the Main Results

### Data Preparation

To reproduce the results in Table 1 of our paper, you can download the following six datasets (Flowers102, RECSIS45, FGVC-Aircraft, CUB, EuroSAT, and DTD) from [HERE](https://www.kaggle.com/datasets/konszer/cpl-datasets). We use the train and test splits provided in the paper [ELEVATER](https://openreview.net/pdf?id=hGl8rsmNXzs) or the dataset suggested splits. 

To facilitate the download process, you can also use the Kaggle API to automatically download all the datasets. 

If you have already set up the Kaggle token and installed the Kaggle API in the Python environment, you can download all the datasets by running the following command:
```bash
kaggle datasets download -d konszer/cpl-datasets
```

Then, before running the experiments, create the following folders to save prompts, datasets, and results:

```bash
mkdir output
mkdir data_
mkdir Features
mkdir script_results
mkdir trained_prompts
```

After downloading and moving the datasets to the `data_` folder, the folder structure should be as follows:
```
data_
├── CUB
│   └── CUB_200_2011
│       ├── ...
│       └── ...
├── DTD
│   ├── ...
│   └── ...
├── EuroSAT
│   ├── ...
│   └── ...
├── FGVCAircraft
│   ├── ...
│   └── ...
├── Flowers102
│   ├── ...
│   └── ...
├── RESICS45
│   ├── ...
│   └── ...
```

### Running the Experiments

To execute the training strategies employing CPL across text prompts and visual prompts, run the following commands:

- For text prompt tuning (including UL, SSL, and TRZSL settings):  
    ```bash
    CUDA_VISIBLE_DEVICES=[...] scripts/run-textPT_ALL.sh "0.02"
    ```

- For visual prompt tuning (including UL, SSL, and TRZSL settings):  
    ```bash
    CUDA_VISIBLE_DEVICES=[...] scripts/run-visualPT_ALL.sh "0.02"
    ```

Replace `[...]` with the GPU number you want to use, and "0.02" represents the value of the default learning rate. Here is an example command:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run-textPT_ALL.sh "0.02"
```

Some notes:
- The running logs from the above scripts are automatically saved in the `output/` directory. 
- The results will be collected by dataset name and saved in `script_results/`. 
- In `trained_prompts/`, the prompt parameters at each iteration and the final prompts after training will be saved. 
- In `Features/`, the features extracted from the images by the vision encoder for each dataset will be pre-saved. This can accelerate the training process of text prompt tuning and is enabled by default. 

For more information about the hyperparameters and settings, you can refer to the comments in the `run-textPT_ALL.sh` and `run-visualPT_ALL.sh` scripts.

**UPDATE**: We now offer a more convenient approach to determine the appropriate hyperparameter $\beta$ for CPL across various settings.

Specifically, we do not directly manipulate $\beta$ as a hyperparameter; instead, we set $\gamma = C \cdot (1 - \beta)$, where $C$ is the number of possible categories and $\gamma$ is the hyperparameter we control, to indirectly regulate the specific value of $\beta$. This approach eliminates the influence of $C$ on $\beta$ and simplifies the process of finding the suitable hyperparameter $\beta$ for different settings, especially settings or datasets with different numbers of categories.

For example, in the code, we can transfer the hyperparameter as `auto*2.0`, which means setting $\gamma = 2.0$. This is equivalent to setting $\beta = 0.2$ for a 10-way classification task.

## Reproducing the Results of CPL+LaFTer

This section of the code is built upon the official [CoOp](https://github.com/KaiyangZhou/CoOp) and [LaFTer](https://github.com/jmiemirza/LaFTer) repositories to reproduce the results in Table 3 of our paper.

### Data Preparation

First, ensure you are operating under the `./LaFTer` directory. All the following commands are executed based on this directory. Navigate to the `./LaFTer` directory by running the following command:

```bash
cd ./LaFTer
```

Next, create the necessary folders under the current directory:
 
```bash
mkdir data
mkdir output
mkdir script_results
```

Download and structure your datasets according to the instructions provided in the [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) official repository. In our paper, we primarily test CPL+LaFTer on the following six datasets: Flowers-102 (oxford_flowers), UCF-101, CIFAR-100, EuroSAT, DTD, and CALTECH-101. These six datasets should be present in the `data/` directory:

```
data
├── caltech-101
├── cifar100
├── dtd
├── eurosat
├── oxford_flowers
├── ucf101
```

Note that CIFAR-100 is not supported by the official CoOp codebase. To download the CIFAR-100 dataset and reproduce the results, we have included the `.png` format CIFAR-100 dataset in the previous Kaggle datasets. If you have already downloaded the datasets from [HERE](https://www.kaggle.com/datasets/konszer/cpl-datasets), you should have the CIFAR-100 dataset. Simply move the CIFAR-100 dataset to the current `data/` folder.

Alternatively, you can download the CIFAR-100 dataset individually from [HERE](https://www.kaggle.com/datasets/konszer/cpl-datasets?select=cifar100).

### Running the Experiments

#### LaFTer with CPL

To execute the training strategies employing CPL within the LaFTer pipeline, run the following command:

```bash
CUDA_VISIBLE_DEVICES=[...] scripts/LaFTer_CPL.sh "0.0005"
```

Replace `[...]` with the specified GPU number you want to use, and "0.0005" is the default learning rate. For example:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/LaFTer_CPL.sh "0.0005"
```

Some notes:
- As with the previous section, the running logs from the above scripts are automatically saved in the `output/` directory. The results will be collected by dataset name and saved in `script_results/`. 
- Note that we employ CPL as an additional module to the original LaFTer pipeline. The implementation of CPL can be found under the `CandidateAPI/` directory, with minor modifications to the original pipeline. This module can guide how to transfer CPL to other Vision-Language Model (VLM) pipelines.

#### LaFTer without CPL

For comparison, you can also run the original LaFTer pipeline as described in their paper by executing the following command:

```bash
CUDA_VISIBLE_DEVICES=[...] scripts/LaFTer.sh "0.0005"
```

[Back to Table of Contents](#table-of-contents)

## Citation 

If you find this work helpful, please consider citing our paper:

```
@inproceedings{zhang2024candidate,
  title = {Candidate Pseudolabel Learning: Enhancing Vision-Language Models by Prompt Tuning with Unlabeled Data},
  shorttitle = {Candidate Pseudolabel Learning},
  booktitle = {Forty-First International Conference on Machine Learning},
  author = {Zhang, Jiahan and Wei, Qi and Liu, Feng and Feng, Lei},
  year = {2024},
  month = jun
}
```