# ExTyDER: Explainable and Typology-Driven Error Reduction for Hierarchical Text Classification

## Overview
This repository introduces **ExTyDER**, an explainable and typology-driven error reduction framework for Hierarchical Text Classification that features:
* **Hierarchy-Aware Error Typology:** A summarized typology that maps misclassifications into specific hierarchical error types.
* **Mitigation Strategies:** A set of proposed baseline strategies tailored to address each identified error type.


---

## Datasets

To run the experiments, you will need to download the datasets used in our evaluation. Please download them from their respective repositories and place the extracted files into the `datasets/` directory under `WebOfScience/` and `FORC/` directories, respectively.

1. **Web of Science (WOS)**
   * **Source:** Mendeley Data
   * **Download Link:** [WOS Dataset (v6)](https://data.mendeley.com/datasets/9rw3vkcfy4/6)

2. **Field of Research Classification (FoRC)**
   * **Source:** Zenodo
   * **Download Link:** [FoRC Dataset](https://zenodo.org/records/10777674)

---

## Usage

### 1. Setup
Make sure you have your environment set up and the required dependencies installed (`pip install -r requirements.txt`). Ensure the WOS and FoRC datasets are downloaded and accessible to the script.

### 2. Running the Framework
You can run the main pipeline by executing `main.py` and specifying your target dataset using the `--dataset` argument. 

To run the pipeline on the **FoRC** dataset:
```bash
python main.py --dataset forc
```
To run the pipeline on the **WOS** dataset:
```bash
python main.py --dataset wos
```
To be concluded all rights reserved.