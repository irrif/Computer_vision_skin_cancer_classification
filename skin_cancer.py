import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import Union

    import marimo as mo

    from IPython.display import display
    from collections import Counter
    import pandas as pd
    import numpy as np

    from sklearn.preprocessing import LabelEncoder

    import torch
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from torchvision.transforms import transforms

    from models import SmallNetwork, ResNet18

    from preprocessing import import_and_preprocess

    import matplotlib.pyplot as plt
    import seaborn as sns

    from PIL import Image

    from datasets import load_dataset
    return (
        Counter,
        DataLoader,
        Dataset,
        F,
        Image,
        LabelEncoder,
        ResNet18,
        SmallNetwork,
        TensorDataset,
        Union,
        display,
        import_and_preprocess,
        load_dataset,
        mo,
        nn,
        np,
        optim,
        pd,
        plt,
        sns,
        torch,
        transforms,
    )


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")
    return (device,)


@app.cell
def _(import_and_preprocess):
    train_dataloader, valid_dataloader, test_dataloader, label_mapping = import_and_preprocess(
        dataset="marmal88/skin_cancer",
        resize=(256, 256),
        centercrop=(224, 224),
        batch_size=64,
        shuffle=True
    )
    return label_mapping, test_dataloader, train_dataloader, valid_dataloader


@app.cell
def _(Counter, label_mapping, plt, sns, train_dataloader):
    unique_labels = train_dataloader.dataset[:][1].tolist()
    label_count = [label_mapping[val] for val in unique_labels]
    counts = Counter(label_count)

    plt.figure(figsize=(12, 6))

    ax = sns.histplot(label_count, bins=len(set(label_count)), discrete=True)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2, height,
                    f"{int(height)}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Distribution of skin cancers")
    ax.set_xlabel("Labels")
    plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()
    return ax, counts, height, label_count, p, unique_labels


@app.cell
def _(display, label_mapping, np, train_dataloader, transforms):
    random_idx = np.random.choice([i for i in range(len(train_dataloader))])
    img = transforms.ToPILImage()(train_dataloader.dataset[random_idx][0])
    img_label = label_mapping[int(train_dataloader.dataset[random_idx][1])]
    print(img_label)
    display(img)
    return img, img_label, random_idx


@app.cell
def _(ResNet18, device, nn, optim, torch, train_dataloader, valid_dataloader):
    # model = SmallNetwork().to(device)
    model = ResNet18(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # train the model for n epochs
    train_loss_history, val_loss_history = [], []
    train_accuracy_history, val_accuracy_history = [], []
    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        train_loss_list = []
        train_correct = 0
        model.train() # Set model in training mode (useful for BatchNorm and Dropout)

        for batch_idx, (data, label) in enumerate(train_dataloader):
            data, label = data.to(device), label.to(device)

            # Make predictions and compute loss
            y_pred = model(data)
            loss = criterion(y_pred, label)
            train_loss_list.append(loss)

            # One step optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Retrieve correct predictions and count them
            train_pred = y_pred.argmax(dim=1, keepdim=True)
            train_correct += train_pred.eq(label.view_as(train_pred)).sum().item()

            # Show metrics foreach 10 batch
            if batch_idx % 10 == 0:
                print("Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                    100 * batch_idx / len(train_dataloader), loss.item()
                ))

        # Average loss and accuracy for the training set
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        train_size = len(train_dataloader.dataset)
        train_accuracy = train_correct / train_size * 100
        print("\nTrain set : Average Loss : {}, Accuracy : {}/{} ({:.0f}%)".format(
            avg_train_loss, train_correct, train_size, train_accuracy
        ))

        train_loss_history.append(avg_train_loss)
        train_accuracy_history.append(train_accuracy)

        # Validation part
        val_loss_list = [] 
        val_correct = 0
        model.eval() # Set the model in evaluation mode
        with torch.no_grad():
            for data, label in valid_dataloader:
                data, label = data.to(device), label.to(device)

                # Predictions and loss calculation
                y_pred = model(data)
                val_loss = criterion(y_pred, label)
                val_loss_list.append(val_loss)

                # Count correct predictions
                pred = y_pred.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(label.view_as(pred)).sum().item()

        # Average loss and accuracy for validation set
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        val_size = len(valid_dataloader.dataset)
        val_accuracy = val_correct / val_size * 100
        print("\nValidation set : Average Loss: {:.4f}, Accuracy : {}/{} ({:.0f}%)\n".format(
            avg_val_loss, val_correct, val_size, val_accuracy
        ))

        val_loss_history.append(avg_val_loss)
        val_accuracy_history.append(val_accuracy)
    return (
        avg_train_loss,
        avg_val_loss,
        batch_idx,
        criterion,
        data,
        epoch,
        label,
        loss,
        model,
        n_epochs,
        optimizer,
        pred,
        train_accuracy,
        train_accuracy_history,
        train_correct,
        train_loss_history,
        train_loss_list,
        train_pred,
        train_size,
        val_accuracy,
        val_accuracy_history,
        val_correct,
        val_loss,
        val_loss_history,
        val_loss_list,
        val_size,
        y_pred,
    )


@app.cell
def _(ResNet18, device, torch):
    model_test = ResNet18(num_classes=7)
    model_test.load_state_dict(torch.load("Models/ResNet18", weights_only=True))
    model_test.to(device)
    model_test.eval()
    return (model_test,)


@app.cell
def _(device, model_test, test_dataloader, torch):
    test_correct = 0
    with torch.no_grad():
        for test_data, test_label in test_dataloader:
            y_pred_test = model_test(test_data.to(device)).argmax(dim=1, keepdim=True)
            test_correct += y_pred_test.eq(test_label.to(device).view_as(y_pred_test)).sum().item()

    print("Test accuracy : {}/{} ({:.0f}%)".format(
        test_correct, len(test_dataloader.dataset), 
        test_correct / len(test_dataloader.dataset) * 100
    ))
    return test_correct, test_data, test_label, y_pred_test


@app.cell
def _(device, label_mapping, model_test, nn, np, plt, test_dataloader, torch):
    idx = np.random.choice(len(test_dataloader.dataset))
    example = test_dataloader.dataset[idx][0].unsqueeze(0).to(device)
    example_label = test_dataloader.dataset[idx][1].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_test(example)
        print(f"Predicted : {output.argmax(1).item()}, Actual : {example_label.item()}")

    probs = nn.functional.softmax(output[0], dim=0).tolist()

    fig, ax_h = plt.subplots(figsize=(10, 5))
    bars = ax_h.barh(list(label_mapping.values()), probs)

    for bar in bars:
        width = bar.get_width()
        ax_h.text(width, bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}", va="center")

    ax_h.set_title("Predicted probabilities per class")
    ax_h.set_xlabel("Probability")
    plt.tight_layout()
    plt.show()
    return (
        ax_h,
        bar,
        bars,
        example,
        example_label,
        fig,
        idx,
        output,
        probs,
        width,
    )


if __name__ == "__main__":
    app.run()
