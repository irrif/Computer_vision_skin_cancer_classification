from models import EarlyStopping

from collections import Counter

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


def plot_diseases_repartition(
        train_loader: DataLoader,
        label_mapping: dict,
    ) -> None:
    """
    Plot the number of occurences for each class.

    Parameters
    ----------
    train_loader : DataLoader
        Train DataLoader object
    label_mapping : dict
        Dictionnary that contains class names as values
    """
    unique_labels = train_loader.dataset.tensors[1].tolist()
    label_count = [label_mapping[val] for val in unique_labels]
    counts = Counter(label_count)

    plt.figure(figsize=(12, 6))

    ax = sns.histplot(label_count, bins=len(set(label_count)), discrete=True)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2, height,
                    f"{int(height)}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Distribution of skin diseases")
    ax.set_xlabel("Labels")
    plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_loss_and_accuracy(
        train_loss_history: list,
        train_accuracy_history: list,
        val_loss_history: list,
        val_accuracy_history: list,
        early_stopping: EarlyStopping,
        n_epochs: int,
    ) -> None:
    """
    Plot loss and accuracy training evolution.

    Parameters
    ----------
    train_loss_history : list
        Training loss for each epoch
    train_accuracy_history : list
        Training accuracy for each epoch
    val_loss_history : list
        Validation loss for each epoch
    val_accuracy_history : list
        Validation accuracy for each epoch
    early_stopping : EarlyStopping
        Early stopping object
    n_epochs : int
        Number of epochs

    Return
    ------
    None
    """
    if not sum(train_loss_history) == 0 and not sum(val_loss_history) == 0:

        if early_stopping.early_stop:
            early_stopping_epoch = early_stopping.epoch_stop
        else:
            early_stopping_epoch = n_epochs

        epochs = range(1, len(train_loss_history) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Loss
        sns.lineplot(x=epochs, y=train_loss_history, label='Train Loss', marker='o', markersize=4, markerfacecolor='black', ax=axes[0])
        sns.lineplot(x=epochs, y=val_loss_history, label='Validation Loss', marker='o', markersize=4, markerfacecolor='black', ax=axes[0])
        axes[0].axvline(x=early_stopping_epoch, color='red', alpha=0.6, linestyle='--', label='Early Stopping')
        axes[0].set_title('Loss over Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot Accuracy
        sns.lineplot(x=epochs, y=train_accuracy_history, label='Train Accuracy', marker='o', markersize=4, markerfacecolor='black', ax=axes[1])
        sns.lineplot(x=epochs, y=val_accuracy_history, label='Validation Accuracy', marker='o', markersize=4, markerfacecolor='black', ax=axes[1])
        axes[1].axvline(x=early_stopping_epoch, color='red', alpha=0.6, linestyle='--', label='Early Stopping')
        axes[1].set_title('Accuracy over Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()


def plot_per_class_metrics(
        metrics: dict,
        label_mapping: dict
    ) -> None:
    """
    Plot, for each class, precision, recall and F1-Score.

    Parameters
    ----------
    metrics : dict
        Dictionnary that contains per class metrics.
    label_mapping: dict
        Dictionnary that contains class names.

    Return
    ------
    None
    """
    data = []
    for cls in range(7):
        data.append({'Disease': f'{label_mapping[cls]}', 'Metric': 'Precision', 'Score': metrics['per_class_precision'][cls]})
        data.append({'Disease': f'{label_mapping[cls]}', 'Metric': 'Recall', 'Score': metrics['per_class_recall'][cls]})
        data.append({'Disease': f'{label_mapping[cls]}', 'Metric': 'F1-Score', 'Score': metrics['per_class_f1'][cls]})

    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(data=df, x='Disease', y='Score', hue='Metric', edgecolor='black', palette='viridis')
    ax.set_axisbelow(True)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='center', padding=3)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)

    plt.title(f"Classification metrics per skin disease\n\
                Overall accuracy ({metrics['overall_accuracy']:.2f}%)")
    plt.legend()
    plt.grid(visible=True, axis='y')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
        y_true: list, 
        y_pred: list, 
        class_names: list = None, 
        normalize: bool = False
    ) -> None:
    """
    Plot a confusion matrix using Seaborn heatmap.

    Parameters:
    y_true : list 
        Ground truth labels.
    y_pred : list 
        Predicted labels.
    class_names : list of str
        Optional list of class names.
    normalize : bool 
        Whether to normalize the matrix row-wise.

    Return
    ------
    None
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    plt.show()
