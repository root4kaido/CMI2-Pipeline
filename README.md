# CMI2-Pipeline

## Setup

For my experiments, I utilized Kaggle Docker. You can simplify the environment setup process by using the latest version of Kaggle Docker. Additionally, you need to install Hydra:

```bash
pip install hydra-core
```

## Structure

```
|-- README.md          # Project overview and setup instructions
|-- config/           
|   |-- inference.yaml # Configuration for inference settings
|   `-- train.yaml     # Configuration for training settings
|-- inference.py       # Script for model inference/prediction
|-- models.py          # Model architecture definitions
|-- notebook/          
|   `-- cmi-pipeline.ipynb  # inference example notebook
|-- preprocess.py      # Data preprocessing module
|-- results/           # Directory for output results
|-- train.py           # Main training script
|-- train_total.sh     # Shell script to run full training pipeline
`-- utils.py           # Utility functions and helper methods
```

## Train
Please edit config/train.yaml and execute the following command:

```bash
python3 train.py
```

Executing the above script will perform training across 10 different random seeds.
The model will be saved as `total_models.pkl` in the directory specified by output_dir. All training results will be logged in the log file. Furthermore, out-of-fold (oof) prediction files for each seed will be generated as CSV files.

## Inference
Please edit config/inference.yaml and execute the following command:

```bash
python3 inference.py
```

The phase parameter determines the inference target:
- `train`: Performs inference on training data entries with existing sii data
- `pseudo`: Performs inference on training data entries where sii data is nan
- `test`: Performs inference on test dataset

Each inference operation generates two output files:
- A .npy file containing the pre-voting ensemble data
- A .csv file containing the post-voting results

The `notebook/cmi-pipeline.ipynb` demonstrates how to execute multiple inferences and perform ensemble operations. Examples of submitting predictions on Kaggle can be found at the link below.
https://www.kaggle.com/code/nomorevotch/cmi-pipeline

## Total Pipeline

I have documented the complete pipeline in `train_total.sh`, which includes initial training, pseudo label inference, and training of four models using these pseudo labels. You can reproduce the final submission by following this script.

```bash
bash train_total.sh
```
