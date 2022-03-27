import random

random.seed(13123123)

# Get random seeds
N_RUNS = 10
RANDOM_SEEDS = random.sample(range(1, 1000000), N_RUNS)

# All parameters definition
PARAMS = {
    # PhysioNet Dataset
    'N_CLASSES': 4,
    'N_CHANNELS': 64,
    'SAMPLING_RATE': 160,
    'N_RUNS': N_RUNS,
    'RANDOM_SEEDS': RANDOM_SEEDS,
    'LABEL_MAP': {'imagine_left_fist': 0, 'imagine_right_fist': 1, 'imagine_both_fist': 2, 'imagine_both_feet': 3},
    'EXCLUSIONS': [88, 89, 92, 100],

    # Global Hyperparameters
    'N_EPOCHS': 2,    
    'LR': 0.001,
    'BATCH_SIZE': 32,
    'SEQ_LEN': 100,
    'SCHEDULER_STEP_SIZE': 10,
    'SCHEDULER_GAMMA': 0.9,
    'TEST_SIZE': 1/10,
    'VALID_SIZE': 1/9,
    
    # FCN hyperparameters
    'FCN_HIDDEN_SIZES': (512, 1024, 512),
    'FCN_DROPOUT_P': 0.4,

    # CNN hyperparameters
    'CNN_HIDDEN_SIZES': (40, 512),
    'CNN_N_KERNELS': 40,
    'CNN_HIDDEN_SIZE': 512,
    'CNN_KERNEL_SIZE': (64, 45),
    'CNN_DROPOUT_P': 0.4,

    # RNN hyperparameters
    'RNN_HIDDEN_SIZE': 256,
    'RNN_N_LAYERS': 2,
    'RNN_DROPOUT_P': 0.4,

    # GCN hyperparameters
    'GCN_HIDDEN_SIZES': (512, 1024, 512),
    'GCN_DROPOUT_P': 0.4,

    # AutoGCN hyperparameters
    'AutoGCN_KERNEL_TYPE': 'b',
    'AutoGCN_HIDDEN_SIZES': (512, 1024, 512),
    'AutoGCN_DROPOUT_P': 0.4,

    # GCRAM hyperparameters
    'GCRAM_GRAPH_TYPE': 'n',
    'GCRAM_CNN_IN_CHANNELS': 1,
    'GCRAM_CNN_N_KERNELS': 40,
    'GCRAM_CNN_KERNEL_SIZE': (64, 45),
    'GCRAM_CNN_STRIDE': 1,
    'GCRAM_DROPOUT1_P': 0.4,
    'GCRAM_LSTM_HIDDEN_SIZE': 64,
    'GCRAM_LSTM_IS_BIDIRECTIONAL': True,
    'GCRAM_LSTM_N_LAYERS': 2,
    'GCRAM_LSTM_DROPOUT_P': 0.4,
    'GCRAM_ATTN_EMBED_DIM': 512,
    'GCRAM_DROPOUT2_P': 0.4,
    'GCRAM_HIDDEN_SIZE': 512,

    # AutoGCRAM hyperparameters
    'AutoGCRAM_GCN_HIDDEN_SIZE': 256,
    'AutoGCRAM_CNN_IN_CHANNELS': 1,
    'AutoGCRAM_CNN_N_KERNELS': 40,
    'AutoGCRAM_CNN_KERNEL_SIZE': (64, 45),
    'AutoGCRAM_CNN_STRIDE': 10,
    'AutoGCRAM_DROPOUT1_P': 0.4,
    'AutoGCRAM_LSTM_HIDDEN_SIZE': 64,
    'AutoGCRAM_LSTM_IS_BIDIRECTIONAL': True,
    'AutoGCRAM_LSTM_N_LAYERS': 2,
    'AutoGCRAM_LSTM_DROPOUT_P': 0.4,
    'AutoGCRAM_ATTN_EMBED_DIM': 512,
    'AutoGCRAM_DROPOUT2_P': 0.4,
    'AutoGCRAM_HIDDEN_SIZE': 512,

    ## TODO
    # GAT hyperparameters
    
    
}   