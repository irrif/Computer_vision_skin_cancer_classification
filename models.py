from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms.functional import adjust_contrast


class SmallNetwork(nn.Module):

    def __init__(self):
        super(SmallNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=0)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=1, padding=0)
        self.act2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=0)
        self.act3 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(in_features=10_368, out_features=6_400)
        self.fc2 = nn.Linear(in_features=6_400, out_features=1_280)
        self.fc3 = nn.Linear(in_features=1_280, out_features=7)


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.pool1(x)

        x = self.conv3(x)
        x = self.act3(x)

        x = self.pool2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

        return output
    

class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1
        ):
        super(BasicBlock, self).__init__()

        # First block convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second block convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Convolution 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # Convolution 2
        out = self.conv2(out)
        out = self.bn2(out)
        # Shortcut connection
        out += self.shortcut(x) 
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    Implement ResNet18 neural network architecture.
    """
    def __init__(
            self,
            num_classes: int = 7
        ):
        super(ResNet18, self).__init__()

        self.num_classes = num_classes
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block=BasicBlock,
            out_channels=64,
            n_blocks=2,
            stride=1
        )

        self.layer2 = self._make_layer(
            block=BasicBlock,
            out_channels=128,
            n_blocks=2,
            stride=2
        )

        self.layer3 = self._make_layer(
            block=BasicBlock,
            out_channels=256,
            n_blocks=2,
            stride=2
        )

        self.layer4 = self._make_layer(
            block=BasicBlock,
            out_channels=512,
            n_blocks=2,
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)


    def _make_layer(self, block, out_channels, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    

    def forward(self, x):
        # ResNet first layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # 4 BasicBlock repeated twice
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Output layer
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    

class EarlyStopping():
    """
    Early stopping class.
    """
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.epoch_stop = 0


    def __call__(
            self,
            val_loss: float,
            model: nn.Module,
            epoch: int
        ) -> None:

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()

        # If score doesn't improve, add 1 to counter
        elif score < self.best_score + self.delta:
            self.counter += 1
            # If counter greater or equal than patience then early stop
            if self.counter >= self.patience:
                self.early_stop = True
                self.epoch_stop = epoch - self.patience

        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0


    def load_best_model(self, model):
        """
        Load best model parameters.
        """
        model.load_state_dict(self.best_model_state)


def train_model(
        model: nn.Module, 
        device: torch.device,
        train_loader: DataLoader,
        loss_function: nn.functional,
        optimizer: torch.optim,
        epoch: int, 
        save: bool, 
        verbose: int
    ) -> Tuple[float]:
    """
    Train a model with the specified optimizer and loss function, over the number of epochs.
    Return loss and accuracy if save=True, otherwise return (0, 0).

    Parameters
    ----------
    model : pytorch model
        model to be trained
    device : torch.device
        Calculation device
    train_loader : torch.DataLoader
        Training DataLoader
    optimizer : torch.optim
        Optimizer
    loss_function : nn.functional
        Loss function
    epoch : int
        Number of epoch to train the model
    save : bool
        Set True to return the loss and accuracy history
    verbose : int
        Print loss and accuracy
        * 1 : For each n batches
        * 2 : For each n batches and for each epoch

    Returns
    -------
    (float, float)

    Example
    -------
    >>> model = ResNet18(num_classes=7)
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> train_dataloader = generate_dataloader(...)
    >>> optimizer = optim.Adam(model.parameters, lr=0.0001)
    >>> criterion = nn.CrossEntropyLoss()
    >>> n_epochs = 50
    >>> train_model(
    >>>     model=model,
    >>>     device=device,
    >>>     train_loader=train_dataloader
    >>>     optimizer=optimizer,
    >>>     loss_function=criterion,
    >>>     save=True,
    >>>     verbose=2
    >>> )
    """
    # Set model in training mode
    model.train()

    losses, corrects = [], 0

    for batch_idx, sample in enumerate(train_loader):
        # Sent data and label to specified device
        data, label = sample['image'].to(device), sample['label'].to(device)
        optimizer.zero_grad() # Set all gradients to 0
        y_pred = model(data)
        loss = loss_function(y_pred, label)
        # Saves batch loss in a list
        if save:
            losses.append(loss)

        loss.backward() # Backpropagation
        optimizer.step()

        # Count correct predictions
        preds = y_pred.argmax(dim=1, keepdim=True)
        corrects += preds.eq(label.view_as(preds)).sum().item()

        # Print loss every x batches
        if verbose >= 1:
            if batch_idx % 10 == 0:
                print("Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100 * batch_idx * len(data) / len(train_loader.dataset), loss.item()
                ))
    # Compute epoch average train loss and train accuracy
    avg_loss = sum(losses) / len(losses)
    overall_accuracy = 100 * corrects / len(train_loader.dataset)

    # Print epoch average loss and accuracy
    if verbose == 2:
        print("\nTrain set : Average loss {:.4f}, Accuracy : {}/{} ({:.0f}%)".format(
            avg_loss, corrects, len(train_loader.dataset), overall_accuracy
        ))

    # Return epoch average loss and accuracy
    if save:
        return avg_loss, overall_accuracy
    else:
        return 0.0, 0.0


def validate_model(
        model: nn.Module, 
        device: torch.device,
        valid_loader: DataLoader,
        loss_function: nn.functional,
        save: bool, 
        verbose: bool
    ) -> Tuple[float]:
    """
    Compute loss and accuracy on validation set.
    Return loss and accuracy if save=True, otherwise return (0, 0).

    Parameters
    ----------
    model : nn.Module
        model to validate
    device : torch.device
        Calculation device
    valid_loader : torch.DataLoader
        Validation DataLoader
    loss_function : nn.functional
        Loss function
    save : bool
        Set True to return the loss and accuracy history
    verbose : bool
        Print loss and accuracy

    Returns
    -------
    (float, float)

        Example
    -------
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> model = ResNet18(num_classes=7).to(device)
    >>> valid_dataloader = generate_dataloader(...)
    >>> optimizer = optim.Adam(model.parameters, lr=0.0001)
    >>> criterion = nn.CrossEntropyLoss()
    >>> validate_model(
    >>>     model=model,
    >>>     device=device,
    >>>     valid_loader=valid_dataloader
    >>>     loss_function=criterion,
    >>>     save=True,
    >>>     verbose=True
    >>> )
    """
    model.to(device)

    val_losses, corrects = [], 0

    model.eval() # Set model in evaluation mode
    with torch.no_grad():
        for sample in valid_loader:
            # Sent data and label to specified device
            data, label = sample['image'].to(device), sample['label'].to(device)

            # Predict and compute loss
            y_pred = model(data)
            loss = loss_function(y_pred, label)

            # Save loss in a list
            if save:
                val_losses.append(loss)

            # Count correct predictions
            pred = y_pred.argmax(dim=1, keepdim=True)
            corrects += pred.eq(label.view_as(pred)).sum().item()

            # Print validation loss and accuracy
            avg_loss = sum(val_losses) / len(val_losses)
            size = len(valid_loader.dataset)
            accuracy = 100 *  corrects / size

        if verbose:
            print("\nValidation set : Average Loss: {:.4f}, Accuracy : {}/{} ({:.0f}%)\n".format(
                avg_loss, corrects, size, accuracy

            ))

    # Return epoch average loss and accuracy
    if save:
        return avg_loss, accuracy
    else:
        return 0.0, 0.0
    

def test_model(
        model: nn.Module,
        device: torch.device,
        test_loader : DataLoader,
        verbose: bool
    ) -> Tuple[float]:
    """
    On test set, return correct per classes, total length of class, overall accuracy.

    Parameters
    ----------
    model : nn.Module
        model to be tested
    device : torch.device
        Calculation device
    test_loader : Dataloader
        Test Dataloader
    verbose : bool
        Set to True to print metrics

    Returns
    -------
    (correct_per_class, total_per_class, overall_accuracy)

    Example
    -------
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> model = ResNet18().to(device)
    >>> test_dataloader = generate_dataloader(...)
    >>> test_model(
    >>>     model=model,
    >>>     device=device,
    >>>     test_loader=test_dataloader,
    >>>     verbose=True
    >>> )
    """
    num_classes = 7
    test_correct = 0
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes
    # Store wrong predictions per true label
    wrong_predictions = [[] for _ in range(num_classes)]

    model.eval()
    with torch.no_grad():
        for sample in test_loader:
            test_data, test_label = sample['image'].to(device), sample['label'].to(device)
            y_pred_test = model(test_data).argmax(dim=1) # prediction
            # Batch correct predictions
            test_correct += y_pred_test.eq(test_label.view_as(y_pred_test)).sum().item()

            # Correct and wrong predictions, per class
            for true_label, pred_label in zip(test_label, y_pred_test):
                true_label = true_label.item()
                pred_label = pred_label.item()
                # Counts the occurrences of each class
                total_per_class[true_label] += 1
                # Count correct predictions per class
                if pred_label == true_label:
                    correct_per_class[true_label] += 1
                # Count wrong predictions per class
                else:
                    wrong_predictions[true_label].append(pred_label)

    # Overall accuracy on test set
    overall_accuracy = test_correct / len(test_loader.dataset) * 100

    if verbose:
        print("Test accuracy : {}/{} ({:.2f}%)".format(
            test_correct, len(test_loader.dataset), 
            overall_accuracy
        ))

    return correct_per_class, total_per_class, overall_accuracy