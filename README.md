# Transformer-based Drowsiness Detection

## Instalation
1. Create virtual env
    ```bash
    conda create -n ml_tdd python==3.10
    ```
2. Install torch. If you are not using GPU (CPU only), install with
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu -y
    ```
    If using GPU, refere to [pytorch get started](https://pytorch.org/get-started/locally/)
3. Install requirements.txt
    ```bash
    pip install -r requirements.txt
    ```

## Training
1. Make sure the data for training and evaluation following the structure below
    ```text
    ├── data_train
    │   └── 1111_1_3_4
    │       └── mmwave_ss.csv
    └── data_train
        └── 1111_1_3_4
            └── mmwave_ss.csv
    ```
    The naming of folder/file containing the data follow {dataset_id}\_{data_session}\_{kss1}\_{kss2}. This is due to the need of extracting kss score from file name
2. Start training 
    ```bash
    python train.py
    ```
3. The best model weight saved at __./model__
