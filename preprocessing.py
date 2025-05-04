from typing import Tuple

import datasets
from datasets import load_dataset

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import transforms
from torchvision.transforms.functional import adjust_contrast

from sklearn.preprocessing import LabelEncoder


class CustomDataset(Dataset):
    """
    Custom Dataset for our skin_cancer hugging face dataset.
    """
    def __init__(
            self,
            tensors: torch.Tensor,
            minority_classes: list,
            train: bool,
            transform: transforms
        ):
        self.tensors = tensors # Data and labels
        self.minority_classes = minority_classes
        self.train = train # train set or not
        self.transform = transform # transformations

    def __len__(self):
        return self.tensors[0].size(0)
    
    def __getitem__(self, idx):
        # Retrieve image
        data = self.tensors[0][idx]
        # Retrieve label
        label = self.tensors[1][idx]
        # Transform image
        if self.transform:
            # If train set, apply data augmentation only on minority classes
            if self.train and label in self.minority_classes:
                data = self.transform(data)
            elif not self.train:
                data = self.transform(data)
        
        # Ensure that all images are 224x224 pixels
        if data.size(2) != 224:
            data = transforms.CenterCrop(size=(224, 224))(data)

        return {"image": data, "label": label}
    

class AdjustContrast:
    """
    Adjust contrast class to be applied during training
    """
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        return adjust_contrast(img, self.contrast_factor)

    
def generate_dataloader(
        dataset: Dataset,
        part_set: str,
        preprocess_transform: transforms,
        label_encoder: LabelEncoder,
        minority_classes: list,
        transform: transforms,
        train: bool,
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
    """
    Parameters
    ----------
    dataset : Dataset
        HuggingFace dataset
    part_set: {'train', 'validation', 'test'}
        Train, validation or test set
    preprocess_transform : list of Transform objects
        List of pre transformations.
    label_encoder : LabelEncoder
        Label encoder object.
    minority_classes : list
        List of under-represented classes (as integers)
    transform : torch.transforms
        Transformation to apply to this specific part set
    train : bool
        True if training set
    batch_size : int
        Batch size
    shuffle : bool
        True to shuffle set during training phase.

    Returns
    -------
    DataLoader

    Example
    -------
    >>> dataset = load_dataset('marmal88/skin_cancer')
    >>> preprocess_transform = transforms.Resize(size=(256, 256))
    >>> label_encoder = LabelEncoder()
    >>> minority_classes = [1, 4, 5]
    >>> train_transform = transforms.Compose([
    >>>     transforms.RandomCrop(size=(224, 224)),
    >>>     transforms.RandomHorizontalFlip(p=0.5)
    >>> ])
    >>> generate_dataloader(
    >>>     dataset=dataset,
    >>>     part_set='train',
    >>>     preprocess_transform=preprocess_transform,
    >>>     label_encoder=label_encoder
    >>>     minority_classes=minority_classes,
    >>>     transform=train_transform,
    >>>     train=True,
    >>>     batch_size=32,
    >>>     shuffle=True
    >>> )
    """
    # Load train, validation or test set
    dataset = load_dataset(dataset, split=part_set)

    # Basic transformations on the dataset
    set_images = import_and_preprocess_image(
        dataset=dataset,
        preprocess_transform=preprocess_transform
    )

    # Extract labels from dataset
    set_labels = extract_labels(
        dataset=dataset,
        label_encoder=label_encoder
    )

    # Create train dataset with data augmentation
    dataset = create_torch_dataset(
        data=set_images,
        label=set_labels,
        part_set=part_set,
        minority_classes=minority_classes,
        train=train,
        transform=transform
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader


def import_and_preprocess_image(
        dataset: datasets.Dataset,
        preprocess_transform: transforms
    ) -> Tuple[torch.Tensor]:
    """
    Transform and preprocess the data.

    Parameters
    ----------
    dataset : datasets.Dataset
        Dataset with data and label as a TensorDataset object
    transformations : list of transforms objects
        List of transformations to be applied to the dataset

    Returns
    -------
    torch.Tensor
    """    
    # If no modifications then transform to tensors
    if preprocess_transform is None:
        preprocess_transform = transforms.ToTensor()
    
    # Preprocess the data in a comprehension list, then turn it into a numpy array and finally in a torch.Tensor
    image_set = image_to_torch(dataset=dataset, transform=preprocess_transform)

    return image_set


def image_to_torch(
        dataset: Dataset,  
        transform: transforms
    ) -> torch.Tensor:
    """
    Converts image dataset to a torch tensor.

    Parameters
    ----------
    dataset : Dataset
        Dataset to transform.
    transform : torch.transforms
        Transformations to apply.

    Returns
    -------
    torch.Tensor

    Example
    -------
    >>> transformations = transforms.Compose([
    >>>                       transforms.Resize(size=(256, 256)),
    >>>                       transforms.ToTensor()
    >>>                   ])
    >>> image_to_torch(dataset, 'train', transformations)
    """
    torch_tensor = torch.from_numpy(
        np.array(
            [transform(dataset[idx]['image']) for idx in range(len(dataset))]
        )
    )

    return torch_tensor


def extract_labels(
        dataset: Dataset,
        label_encoder: LabelEncoder
    ) -> torch.Tensor:
    """
    Extract labels from the dataset and convert them into numerical values.
    Returns labels encoded as torch.Tensor

    Parameters
    ----------
    dataset : HuggingFace Dataset
        Dataset containing labels.
    label_encoder : LabelEncoder
        Label encoder object.

    Returns
    -------
    torch.Tensor
    """

    # If label encoder has already been fitted
    if not hasattr(label_encoder, 'classes_'):
        labels = torch.from_numpy(label_encoder.fit_transform(np.array(dataset['dx'])))
    # If label encoder first time met labels
    else:
        labels = torch.from_numpy(label_encoder.transform(np.array(dataset['dx'])))

    return labels


def create_torch_dataset(
        data: torch.Tensor,
        label: torch.Tensor,
        part_set: str,
        minority_classes: list,
        train: bool,
        transform: transforms
    ) -> Dataset:
    """
    Create a pytorch dataset with possible transformations.

    Parameters
    ----------
    data : torch.Tensor
        Images
    label : torch.Tensor
        Corresponding labels
    part_set : {'train', 'validation', 'test'}
        Training, validation or test set
    minority_classes : list
        List of under-represented classes
    train : bool
        Set to True if training set
    transform : transforms
        Transformations to apply
        
    Returns
    -------
    Dataset
    """
    if part_set == 'train':
        train = True
    else:
        train = False

    dataset = CustomDataset(
        tensors=(data, label),
        minority_classes=minority_classes,
        train=train,
        transform=transform
    )

    return dataset


def create_dataloader(dataset: Dataset = None, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
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

    Example
    -------
    >>> le = LabelEncoder()
    >>> labels = ['label1', 'label2', 'label3']
    >>> le.fit_transform(labels)
    >>> get_labels_mapping(labelencoder=le)
    {0: 'label1', 1: 'label2', 2: 'label3'}
    """

    return dict(zip(labelencoder.transform(labelencoder.classes_), labelencoder.classes_))