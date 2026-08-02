# DATASET VARIAABLES
INPUT_CHUNK_LEN = 60 # 2 MINUTES
OUTPUT_CHUNK_LEN = 10 # 10 SECONDS
WINDOW_STRIDE = 0.9 * INPUT_CHUNK_LEN #90% # offset between the beginning of each window
GAP = 0 # offset between context and prediction

# DARTS TFT MODEL VARIABLES
BATCH_SIZE = 128
HIDDEN_SIZE = 32 
LSTM_LAYERS = 2
ATT_HEADS = 4
DROPOUT = 0.1

# OTHER MODEL VARIABLES
N_LAYERS = 2 #  optional
D_MODEL = 16 #64 / 128 optional
N_HEADS = 2 # 8/16 optional

# TRAINING VARIABLES
BATCH_SIZE = 128
DROPOUT = 0.3
L_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCH = 5
RANDOM_SEED = 42
SAMPLE_PER_TS = 1000 # change later after calculating based on input and output optimal chunk

# TEST VARIABLES
PRED_ITERATE = 50