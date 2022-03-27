import os
import time
import random
import copy

import numpy as np
import torch
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split

from params import PARAMS
from utils import load_data


def get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, random_seed, device):
    """
    Create dataloaders for train, valid, test datasets respectively
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    X_train, y_train = torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)
    X_valid, y_valid = torch.tensor(X_valid).to(device), torch.tensor(y_valid).to(device)
    X_test, y_test = torch.tensor(X_test).to(device), torch.tensor(y_test).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    dataloaders = {'train': train_loader, 'val': valid_loader, 'test': test_loader}
    return dataloaders

def prepare_datasets(random_seed, dataset_name, output_path, device):
    """
    Load pickle datasets, split to train, valid, test sets then create dataloaders for each set
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    dataset_path = os.path.join('./dataset/train', dataset_name + '.pickle')
    X_train, y_train, X_test, y_test = load_data(dataset_path)

    """
    Instead of using separate subjects for train and test datsets, 
    mix all data then random pick train, valid test datasets
    """ 

    # Add channel
    if 'cnn' in output_path:
        X_train, X_test = np.expand_dims(X_train, 1), np.expand_dims(X_test, 1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=PARAMS['VALID_SIZE'], shuffle=False, random_state=random_seed)
    
    dataloaders = get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, PARAMS['BATCH_SIZE'], random_seed=random_seed, device=device)

    return dataloaders


def init_model_params(model, random_seed):
    """
    Initialize model weights and biases manually for random seeds.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.uniform_(p)
    return model

# # """
# # This func copied from 
# # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# # and slightly edited

# # WARNING: Training works fine but printed accuracy and loss are wrong!!!
# # """

# def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs, random_seed, device):
#     random.seed(random_seed)
#     np.random.seed(random_seed)
#     torch.manual_seed(random_seed)
#     torch.cuda.manual_seed(random_seed)

#     since = time.time()

#     # For saving history
#     train_loss = torch.zeros(num_epochs)
#     val_loss = torch.zeros(num_epochs)
#     train_acc = torch.zeros(num_epochs)
#     val_acc = torch.zeros(num_epochs)
#     lrs = []

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs-1}')
#         print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0.0

#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             # Save history
#             if phase == 'train':
#                 train_acc[epoch] = epoch_acc
#                 train_loss[epoch] = epoch_loss
#                 lrs.append(scheduler.get_last_lr()[0])
#             elif phase == 'val':
#                 val_acc[epoch] = epoch_acc
#                 val_loss[epoch] = epoch_loss

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#         print()

#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed%60:.0f}s')

#     return model, {'train_loss': train_loss.tolist(), 'train_acc': train_acc.tolist(), 'val_loss': val_loss.tolist(), 'val_acc': val_acc.tolist(), 'lrs': lrs, 'train_time': time_elapsed}


"""
This function is copied from:
https://inside-machinelearning.com/en/the-ideal-pytorch-function-to-train-your-model-easily/#The_training_function
This function is overall easy to understand than function above.
"""
def train_model_2(model, optimizer, scheduler, loss_fn, train_dl, val_dl, epochs, random_seed, device, verbose=False):
    """
    This function trains the model then return model with best accuracy and training history.
    ...
    Parameters
    ----------
    model : torch.nn.Module Object
        Model to train.
    optimizer : torch.optim.Optimizer Object
        Optimizer for model.
    scheduler : 
        Learning rate scheduler for the optimizer
    loss_fn : 
        Loss function/criterai for the model
    train_dl : torch.utils.data.DataLoader object
        Training set dataloader
    val_dl : torch.utils.data.DataLoader object
        Validation set dataloader
    epochs : int
        Number of epochs to train
    random_seed : int
        Random seed
    device : torch.device Object
        Device to train
    verbose : bool
        Set it to True to print training time per epoch and val accuracy

    Returns
    -------
    model : torch.nn
        Trained model with best accuracy
    history : dict
        Training history (loss, accuracy)
    """

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # Initlize best model and acc
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if verbose:
        print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    start_time_sec = time.time()

    for epoch in range(1, epochs+1):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:

            optimizer.zero_grad()

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)

        # Run scheduler on every 10 epochs
        if epoch % 10:
            scheduler.step()

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)

        # If val accuracy is better than before than save it as best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 10 == 0:
            if verbose:
                print('Epoch %3d/%3d, LR: %5.4f, train loss: %5.4f, train acc: %5.4f, val loss: %5.4f, val acc: %5.4f' % \
                (epoch, epochs, scheduler.get_lr()[-1], train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs

    if verbose:
        print()
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
        print('Best val Acc: {:4f}'.format(best_acc))

    # Load params with highest accuracy
    model.load_state_dict(best_model_wts)

    return model, history

def model_predict(model, test_loader, device):
    """
    Predict data in dataloader using model
    """
    # Set model to eval mode
    model.eval()

    # Predict without computing gradients
    with torch.no_grad():
        y_preds = []
        y_true = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_preds.append(preds)
            y_true.append(labels)

    y_preds = torch.cat(y_preds).tolist()
    y_true = torch.cat(y_true).tolist()

    return y_preds, y_true