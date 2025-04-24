from typing import Tuple

from datasets import load_dataset

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchvision.transforms import transforms

from sklearn.preprocessing import LabelEncoder


def import_and_preprocess(dataset: str = "marmal88/skin_cancer",
                          resize: tuple = (256, 256),
                          centercrop: tuple = (224, 224),
                          batch_size: int = 32,
                          shuffle: bool = True) -> Tuple[DataLoader]:
    """
    Import and preprocess the dataset by resizing it, croping it and convert it to a tensor.
    Returns training, validation and test set as a tensor of shape (n, 3, 224, 224) + label mapping as a dict.

    Parameters
    ----------
    resize : tuple
        Size at which to resize the photo
    centercrop : tuple
        Desired output size of the crop
    batch_size : int
        Batch size
    shuffle : bool
        Set to True to have the data reshuffled at every epoch.

    Returns
    -------
    tuple of DataLoader + dict
    """
    # Import dataset
    ds = load_dataset(dataset)

    # Preprocess and resize the data
    train_data, valid_data, test_data = transform_and_preprocess(
        dataset=ds, resize=resize, centercrop=centercrop
    )

    # Extract labels
    train_labels, valid_labels, test_labels, mapping_dict = extract_labels(
        dataset=ds
    )
    
    # Create TensorDataset objects
    train_dataset = create_tensor_dataset(train_data, train_labels)
    valid_dataset = create_tensor_dataset(valid_data, valid_labels)
    test_dataset = create_tensor_dataset(test_data, test_labels)

    # Create DataLoader objects
    train_dataloader = create_dataloader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = create_dataloader(dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = create_dataloader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, valid_dataloader, test_dataloader, mapping_dict


def transform_and_preprocess(dataset,
                             resize: tuple = (256, 256),
                             centercrop: tuple = (224, 224),
                             transformations = None) -> Tuple[torch.Tensor]:
    """
    Transform and preprocess the data.

    Parameters
    ----------
    dataset : TensorDataset
        Dataset with data and label as a TensorDataset object
    resize : tuple
        Size at which to resize the photo
    centercrop : tuple
        Desired output size of the crop
    transformations : list of Transform objects
        List of transformations to be applied to the dataset

    Returns
    -------
    tuple of torch.Tensor
    """

    if not transformations:
        preprocess_transforms = transforms.Compose([
            transforms.Resize(size=resize),
            transforms.CenterCrop(size=centercrop),
            transforms.ToTensor()
        ])
    else:
        preprocess_transforms = transformations
    
    # Preprocess the data in a comprehension list, then turn it into a numpy array and finally in a torch.Tensor
    train_orig = torch.from_numpy(
        np.array([
            preprocess_transforms(dataset['train'][idx]['image']) for idx in range(len(dataset['train']))
        ])
    )

    valid_orig = torch.from_numpy(
        np.array([
            preprocess_transforms(dataset['validation'][idx]['image']) for idx in range(len(dataset['validation']))
        ])
    )

    test_orig = torch.from_numpy(
        np.array([
            preprocess_transforms(dataset['test'][idx]['image']) for idx in range(len(dataset['test']))
        ])
    )

    return train_orig, valid_orig, test_orig


def extract_labels(dataset) -> tuple[torch.Tensor]:
    """
    Extract labels from the dataset and convert them into numerical values.
    Returns training, validation and test label as a Tensor, and label mapping as a dict.

    Parameters
    ----------
    dataset : pd.DataFrame or Dataset
        Dataset containing labels

    Returns
    -------
    tuple of torch.Tensor and dict
    """

    le = LabelEncoder()

    train_labels = torch.from_numpy(le.fit_transform(np.array(dataset['train']['dx'])))
    valid_labels = torch.from_numpy(le.transform(np.array(dataset['validation']['dx'])))
    test_labels = torch.from_numpy(le.transform(np.array(dataset['test']['dx'])))

    mapping_dict = get_labels_mapping(labelencoder=le)

    return train_labels, valid_labels, test_labels, mapping_dict


def get_labels_mapping(labelencoder: LabelEncoder) -> dict:
    """
    Return a dictionnary containing the mapping between labels and tags.
    Key is an integer and value a string.

    Parameters
    ----------
    labelencoder : LabelEncoder
        Scikit-learn LabelEncoder

    Returns
    -------
    dict
    """

    return dict(zip(labelencoder.transform(labelencoder.classes_), labelencoder.classes_))


def create_tensor_dataset(data: torch.Tensor = None, labels: torch.Tensor = None) -> TensorDataset:
    """
    Create a TensorDataset object.

    Parameters
    ----------
    data : torch.Tensor
        Dataset as a Tensor object
    labels : torch.Tensor
        Labels as a Tensor object

    Returns
    -------
    TensorDataset
    """

    dataset = TensorDataset(data, labels)

    return dataset


def create_dataloader(dataset: TensorDataset = None, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader object.

    Parameters
    ----------
    dataset : TensorDataset
        Dataset with data + labels, as a TensorDataset object
    batch_size : int
        Batch size
    shuffle : bool
        Set to True to have the data reshuffled at every epoch.

    Returns
    -------
    DataLoader
    """

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader