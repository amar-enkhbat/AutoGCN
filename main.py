import warnings
warnings.filterwarnings("ignore")

# import utilites libraries
import os
import random
from datetime import datetime
import pickle
from tqdm import tqdm

# Import computation libraries
import numpy as np
import torch

# Import libraries from local files
from utils import print_classification_report, plot_history, plot_cm, plot_adj, show_metrics
from models import FCN, CNN, RNN, GCN, AutoGCN, AutoGCRAM, GCRAM
from params import PARAMS
from train import prepare_datasets, train_model_2, init_model_params, model_predict


def run_model(random_seed, dataloaders, model, output_path, device):
    """
    Run model and save evaluation results such as classification results, training history, confusion matrix etc 

    Parameters
    ----------
    random_seed : int
        Random seed
    dataloaders : dict
        dict with train, test PyTorch dataloader objects
    model : torch.nn.Module
        PyTorch model to run

    """
    # Define random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # Define model params    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['LR'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, PARAMS['SCHEDULER_STEP_SIZE'], PARAMS['SCHEDULER_GAMMA'])

    # Train model
    best_model, history = train_model_2(model, optimizer, scheduler, criterion, dataloaders['train'], dataloaders['val'], PARAMS['N_EPOCHS'], random_seed, device)
    # Use best model for evaluation
    best_model = best_model.to(device)

    # Predict test dataset
    y_preds, y_test = model_predict(best_model, test_loader=dataloaders['test'], device=device)

    # Save training history
    cr, cm, auroc = print_classification_report(y_test, y_preds, list(PARAMS['LABEL_MAP'].keys()))
    plot_history(history, output_path)

    # Save confusion matrices
    plot_cm(cm, list(PARAMS['LABEL_MAP'].keys()), output_path)
    if 'gcn' in output_path or 'auto' in output_path:
        plot_adj(best_model.adj.cpu().detach().numpy(), f'{output_path}/trained_adj.png')
        pickle.dump(best_model.adj.cpu().detach().numpy(), open(f'{output_path}/trained_adj.pickle', 'wb'))
    
    # Save results
    n_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    results = {'history': history, 'cm': cm.tolist(), 'cr': cr,'auroc': auroc , 'n_params': n_params}
    with open(os.path.join(output_path, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f)


def model_picker(model_name, random_seed, output_path, device):
    """
    Pick models between FCN, CNN, RNN, GCN, AutoGCN, GCRAM, AutoGCRAM then initialize parameters.
    Trainable adjacency matrices of AutoGCN and AutoGCRAM models are initialized same as othes but their diagonal values are set to 1 to let models focus on respective channels initially.
    """
    if model_name == 'fcn':
        model = FCN(in_features=PARAMS['SEQ_LEN'], 
        num_classes=PARAMS['N_CLASSES'], 
        n_nodes=PARAMS['N_CHANNELS'], 
        hidden_sizes=PARAMS['FCN_HIDDEN_SIZES'], 
        dropout_p=PARAMS['FCN_DROPOUT_P'])

    elif model_name == 'cnn':
        model = CNN(PARAMS['CNN_KERNEL_SIZE'], 
        PARAMS['SEQ_LEN'], 
        PARAMS['CNN_N_KERNELS'], 
        PARAMS['CNN_HIDDEN_SIZE'], 
        PARAMS['N_CLASSES'], 
        dropout_p=PARAMS['CNN_DROPOUT_P'])

    elif model_name == 'rnn':
        model = RNN(PARAMS['SEQ_LEN'], 
        PARAMS['RNN_N_LAYERS'], 
        PARAMS['RNN_HIDDEN_SIZE'], 
        PARAMS['N_CLASSES'], 
        dropout_p=PARAMS['RNN_DROPOUT_P'])

    elif model_name == 'gcn':
        model = GCN(in_features=PARAMS['SEQ_LEN'], 
        n_nodes=PARAMS['N_CHANNELS'], 
        num_classes=PARAMS['N_CLASSES'], 
        hidden_sizes=PARAMS['GCN_HIDDEN_SIZES'], 
        graph_type='n', 
        dropout_p=PARAMS['GCN_DROPOUT_P'], 
        device=device)

    elif model_name == 'gcn_auto':
        model = AutoGCN(kernel_type=PARAMS['AutoGCN_KERNEL_TYPE'],
        in_features=PARAMS['SEQ_LEN'], 
        n_nodes=PARAMS['N_CHANNELS'], 
        num_classes=PARAMS['N_CLASSES'], 
        hidden_sizes=PARAMS['AutoGCN_HIDDEN_SIZES'], 
        dropout_p=PARAMS['AutoGCN_DROPOUT_P'], 
        device=device)

    elif model_name == 'gcram':
        model = GCRAM(graph_type=PARAMS['GCRAM_GRAPH_TYPE'], 
        seq_len=PARAMS['SEQ_LEN'], 
        cnn_in_channels=PARAMS['GCRAM_CNN_IN_CHANNELS'], 
        cnn_n_kernels=PARAMS['GCRAM_CNN_N_KERNELS'], 
        cnn_kernel_size=PARAMS['GCRAM_CNN_KERNEL_SIZE'], 
        cnn_stride=PARAMS['GCRAM_CNN_STRIDE'], 
        lstm_hidden_size=PARAMS['GCRAM_LSTM_HIDDEN_SIZE'], 
        is_bidirectional=PARAMS['GCRAM_LSTM_IS_BIDIRECTIONAL'], 
        lstm_n_layers=PARAMS['GCRAM_LSTM_N_LAYERS'], 
        attn_embed_dim=PARAMS['GCRAM_ATTN_EMBED_DIM'], 
        n_classes=PARAMS['N_CLASSES'], 
        lstm_dropout_p=PARAMS['GCRAM_LSTM_DROPOUT_P'], 
        dropout1_p=PARAMS['GCRAM_DROPOUT1_P'], 
        dropout2_p=PARAMS['GCRAM_DROPOUT2_P'], 
        device=device)

    elif model_name == 'gcram_auto':
        model = AutoGCRAM(seq_len=PARAMS['SEQ_LEN'], 
        n_nodes=PARAMS['N_CHANNELS'],
        gcn_hidden_size=PARAMS['AutoGCRAM_GCN_HIDDEN_SIZE'],
        cnn_in_channels=PARAMS['AutoGCRAM_CNN_IN_CHANNELS'], 
        cnn_n_kernels=PARAMS['AutoGCRAM_CNN_N_KERNELS'], 
        cnn_kernel_size=PARAMS['AutoGCRAM_CNN_KERNEL_SIZE'], 
        cnn_stride=PARAMS['AutoGCRAM_CNN_STRIDE'], 
        lstm_hidden_size=PARAMS['AutoGCRAM_LSTM_HIDDEN_SIZE'], 
        is_bidirectional=PARAMS['AutoGCRAM_LSTM_IS_BIDIRECTIONAL'], 
        lstm_n_layers=PARAMS['AutoGCRAM_LSTM_N_LAYERS'], 
        attn_embed_dim=PARAMS['AutoGCRAM_ATTN_EMBED_DIM'], 
        n_classes=PARAMS['N_CLASSES'], 
        lstm_dropout_p=PARAMS['AutoGCRAM_LSTM_DROPOUT_P'], 
        dropout1_p=PARAMS['AutoGCRAM_DROPOUT1_P'], 
        dropout2_p=PARAMS['AutoGCRAM_DROPOUT2_P'], 
        device=device)
    
    # Initialize models
    model = init_model_params(model, random_seed=random_seed)
    
    # Set adjacency matrix diagonals to 1 if model is AutoGCN or AutoGCRAM. Save initialized adjacency matrix.
    if 'auto' in model_name:
        model.init_adj_diag()
        if output_path is not None:
            pickle.dump(model.adj.cpu().detach().numpy(), open(f'{output_path}/untrained_adj.pickle', 'wb'))
            plot_adj(model.adj.cpu().detach().numpy(), f'{output_path}/untrained_adj.png')

    model = model.to(device)

    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Models to train
    model_names = ['fcn', 'cnn', 'rnn', 'gcn', 'gcn_auto', 'gcram', 'gcram_auto']

    # Datasets to train
    n_subjects = 20
    # n_subjects = 50
    # n_subjects = 105
    dataset_names = [f'cross_subject_data_{i}_{n_subjects}_subjects' for i in range(PARAMS['N_RUNS'])]

    # Only use 1 random seed due to computation time
    random_seeds = PARAMS['RANDOM_SEEDS'][:1]

    print('#' * 50)
    print('Model names:', model_names)
    print('Number of models:', len(model_names))
    print('Dataset names:', dataset_names)
    print('Number of datasets:', len(dataset_names))
    print('Random seeds:', random_seeds)
    print('Number of random seeds:', len(random_seeds))
    print('#' * 50)
    print('PARAMS:')
    print(PARAMS)
    print('')
    
    input_key = input('Execute Y/N?  ')

    if input_key in 'nN':
        exit(1)

    # Start training
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    for model_name in tqdm(model_names):
        for dataset_name in dataset_names:
            for random_seed in random_seeds:
                # Define output path
                output_path = os.path.join('output', time_now, model_name, dataset_name, str(random_seed))
                os.makedirs(output_path, exist_ok=True)
                with open(os.path.join('output', time_now, 'params.txt'), 'w') as f:
                    f.write(str(PARAMS))

                # Load datasets
                dataloaders = prepare_datasets(random_seed, dataset_name, output_path, device)

                # Pick model
                model = model_picker(model_name, random_seed, output_path, device)
                
                # Train model
                run_model(random_seed, dataloaders, model, output_path, device)

    final_results = show_metrics(time_now, model_names, dataset_names, random_seeds)

if __name__=='__main__':
    main()