# Transformer-based Drowsiness Detection
## Environment Preparation
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
4. We used optuna and optuna-dashboard to do hyperparameter tunning. It is not required, but if you want to visualize the tunning result, install optima-dashboard
    ```bash
    pip install optuna-dashboard
    ```

## Hyperparameter Tunning
The hyperparameter tunning iss evaluated based on **RMSE**. We are using optuna to automate this task. The default configuration is below:
```text
data = mmwave_ss.csv # we treat file name as peripheral name
n_trials = 50
```
Start tunning by
```bash
python h_tunning.py
```
To visualize tunning results using optuna-dashboard, run
```bash
optuna-dashboard sqlite:///ml_tft.db
```
## Training
1. Make sure the data for training, evaluation, and testing following the structure below
    ```text
    ├── data_train
    │   └── 1111_1_3_4
    │       └── mmwave_ss.csv
    ├── data_val
    │   └── 1111_1_3_4
    │       └── mmwave_ss.csv
    └── data_test
        └── 1111_1_3_4
            └── mmwave_ss.csv
    ```
    The naming of folder/file containing the data follow {dataset_id}\_{data_session}\_{kss1}\_{kss2}. This is due to the need of extracting kss score from file name
2. Start training by
    ```bash
    python train.py
    ```
3. The best model weight saved at __./model__

## Testing
Please not, that testing also follow the same data structure
```text
└── data_test
    └── 1111_1_3_4
        └── mmwave_ss.csv
```