import os
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


def load_data(dataset_path):
    """
    Load train, test data from pickle file
    """
    dataset = pickle.load(open(dataset_path, 'rb'))
    y_train = dataset['y_train']
    X_train = dataset['X_train'].astype(np.float32)
    y_test = dataset['y_test']
    X_test = dataset['X_test'].astype(np.float32)

    return X_train, y_train, X_test, y_test


def print_classification_report(y_true, y_preds, class_names):
    """
    Compute classification report
    """
    num_classes = len(class_names)

    # Calculate accuracy, precision, recall etc
    cr = classification_report(y_true, y_preds, output_dict=True, target_names=class_names)
    cm = confusion_matrix(y_true, y_preds)

    # One hot encode y_preds
    y_preds_ohe = np.zeros((len(y_preds), num_classes))
    for i, j in enumerate(y_preds):
        y_preds_ohe[i, j] = 1

    # One hot encode y_true
    y_true_ohe = np.zeros((len(y_true), num_classes))
    for i, j in enumerate(y_true):
        y_true_ohe[i, j] = 1

    # Calculate ROC AUC
    auroc = roc_auc_score(y_true_ohe, y_preds_ohe, multi_class='ovo')

    return cr, cm, auroc


def plot_history(history, save_path):
    """
    Plot loss and accuracy then save as html file.
    """
    df = pd.DataFrame(history)
    df["Epochs"] = range(len(df))
    
    fig = px.line(df, x='Epochs', y=['loss', 'val_loss'], labels={'value': 'Loss'})
    fig.write_html(os.path.join(save_path, 'history_loss.html'))

    fig = px.line(df, x='Epochs', y=['acc', 'val_acc'], labels={'value': 'Loss'})
    fig.write_html(os.path.join(save_path, 'history_acc.html'))


def plot_cm(cm, class_names, save_path):
    """
    Plot confusion matrix as image.
    """
    plt.figure(figsize=(7, 5))
    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
    # cm_df = pd.DataFrame(cm)
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cm.png'))
    # plt.show()
    plt.clf()
    plt.close()

def plot_adj(adj, save_path):
    """
    Plot adjacency matrix as image
    """
    plt.figure(figsize=(7, 5))
    sns.heatmap(adj, fmt='g')
    plt.ylabel('Features')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.clf()
    plt.close()


def show_metrics(time_now, model_names, dataset_names, random_seeds):
    """Save model metrics per run, per dataset and per model"""
    final_results = []
    for model_name in model_names:
        for dataset_name in dataset_names:
            subject_idc = json.load(open(os.path.join('./dataset/train', dataset_name + '.json'), 'r'))
            accs = []
            precs_macro = []
            precs_weighted = []
            recalls_macro = []
            recalls_weighted = []
            aurocs = []
            n_trainable_params = []

            for random_seed in random_seeds:
                path = os.path.join('output', time_now, model_name, dataset_name.rstrip('.pickle').lstrip('./dataset/train/'), str(random_seed), 'results.pickle')
                results = pickle.load(open(path, 'rb'))

                accs.append(results['cr']['accuracy'])
                precs_macro.append(results['cr']['macro avg']['precision'])
                precs_weighted.append(results['cr']['weighted avg']['precision'])
                recalls_macro.append(results['cr']['macro avg']['recall'])
                recalls_weighted.append(results['cr']['weighted avg']['recall'])
                aurocs.append(results['auroc'])
                n_trainable_params.append(results['n_params'])
            result_df = pd.DataFrame({'model_name': [model_name for i in random_seeds], 'dataset_name': [dataset_name for i in random_seeds],'train_idc': [subject_idc['train_idc'] for i in random_seeds], 'test_idc': [subject_idc['test_idc'] for i in random_seeds], 'random_seed': random_seeds, 'accuracy': accs, 'precision_macro': precs_macro, 'precision_weighted': precs_weighted, 'recall_macro': recalls_macro, 'recall_weighted': recalls_weighted, 'AUROC': aurocs, 'n_params': n_trainable_params})
            final_results.append(result_df)
    final_results = pd.concat(final_results).reset_index(drop=True)
    final_results.to_csv(os.path.join('./output', time_now, 'results.csv'), index=False)

    std_per_dataset = final_results.groupby(['model_name', 'dataset_name']).std().drop(columns=['random_seed'])
    std_per_dataset = std_per_dataset.rename(columns={'accuracy': 'accuracy_std',
                        'precision_macro': 'precision_macro_std', 
                        'precision_weighted': 'precision_weighted_std', 
                        'recall_macro': 'recall_macro_std', 
                        'recall_weighted': 'recall_weighted_std'})
    std_per_dataset = std_per_dataset.drop(columns=['AUROC', 'n_params'])
    mean_per_dataset = final_results.groupby(['model_name', 'dataset_name']).mean().drop(columns='random_seed')
    results_per_dataset = pd.concat([mean_per_dataset, std_per_dataset], axis=1)
    results_per_dataset.to_csv(os.path.join('./output', time_now, 'results_per_dataset.csv'))

    std_per_model = final_results.groupby(['model_name']).std().drop(columns=['random_seed'])
    std_per_model = std_per_model.rename(columns={'accuracy': 'accuracy_std',
                        'precision_macro': 'precision_macro_std', 
                        'precision_weighted': 'precision_weighted_std', 
                        'recall_macro': 'recall_macro_std', 
                        'recall_weighted': 'recall_weighted_std'})
    std_per_model = std_per_model.drop(columns=['AUROC', 'n_params'])
    mean_per_model = final_results.groupby(['model_name']).mean().drop(columns='random_seed')
    results_per_model = pd.concat([mean_per_model, std_per_model], axis=1)
    results_per_model.to_csv(os.path.join('./output', time_now, 'results_per_model.csv'))
    print(results_per_model)
    return final_results