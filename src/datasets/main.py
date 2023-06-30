from .SAVEE import SAVEEDataset
from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .text import TextDataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'text', 'savee')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'text':
        dataset = TextDataset(filepath=data_path) # no need for the class as we only have one - sport

    if dataset_name == 'savee':
        dataset = SAVEEDataset(root=data_path)

    return dataset
