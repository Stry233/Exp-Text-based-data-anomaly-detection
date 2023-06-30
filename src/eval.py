import logging
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import click
import torch
import pandas as pd

from deepSVDD import DeepSVDD
from datasets.main import load_dataset


def calculate_label_score(data, deepSVDD):
    """
    Calculate labels and scores for given data.

    Parameters:
    data (Tuple[torch.Tensor]): Tuple of inputs, labels and indices from the DataLoader.
        inputs (torch.Tensor): Input data.
        labels (torch.Tensor): Ground truth labels.
        idx (torch.Tensor): Indices of the data.
    deepSVDD (DeepSVDD): The neural network model to use for prediction.

    Returns:
    List[Tuple[int, int, float]]: List of tuples with indices, labels and calculated scores.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    R = torch.tensor(deepSVDD.R, device=device)  # radius R initialized with 0 by default.
    c = torch.tensor(deepSVDD.c, device=device) if deepSVDD.c is not None else None
    nu = deepSVDD.nu

    inputs, labels, idx = data
    inputs = inputs.to(device)
    net = deepSVDD.net.to(device)
    outputs = net(inputs)
    dist = torch.sum((outputs - c) ** 2, dim=1)

    if deepSVDD.objective == 'soft-boundary':
        scores = dist - R ** 2
    else:
        scores = dist

    # Save triples of (idx, label, score) in a list
    idx_label_score = list(zip(idx.cpu().data.numpy().tolist(),
                               labels.cpu().data.numpy().tolist(),
                               scores.cpu().data.numpy().tolist()))

    return idx_label_score


@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'text']), default='text')
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'text_ResNet']),
                default='text_ResNet')
@click.argument('load_model', type=click.Path(exists=True), default='../log/text/model.tar')
@click.argument('data_path', type=click.Path(exists=True), default='../data/twitter/smaller_train.csv')
@click.argument('save_path', type=click.Path(), default='../score')
# @click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, load_model, data_path, save_path, normal_class):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = save_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD('one-class', 0.1)
    deep_SVDD.set_network(net_name)

    # Load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    deep_SVDD.load_model(model_path=load_model, load_ae=False)

    # Get train data loader
    train_loader, _ = dataset.loaders(batch_size=1)

    logger.info('Start evaluation...')
    all_scores = []
    with torch.no_grad():
        for data in train_loader:
            scores = calculate_label_score(data, deep_SVDD)
            all_scores.extend(scores)
    df_scores = pd.DataFrame(all_scores, columns=['Index', 'Label', 'Score'])
    # Calculate scores for the entire CIFAR10 dataset
    if dataset_name == 'cifar10':
        # Save scores to a CSV file
        df_scores.to_csv(save_path + '/outlier_scores.csv', index=False)
        logger.info('Anomaly scores saved to %s.' % (save_path + '/outlier_scores.csv'))
    elif dataset_name == 'text':
        normal_score = [(x - min(df_scores['Score'])) / (max(df_scores['Score']) - min(df_scores['Score'])) for x in df_scores['Score']]
        print(roc_auc_score(df_scores['Label'], normal_score))
        print([x for x in zip(df_scores['Label'], normal_score)])

        # Convert probabilities to binary predictions with 0.5 as threshold
        predicted = [1 if prob >= 0.5 else 0 for prob in normal_score]

        # generate filtered data
        df = pd.read_csv(data_path)
        df_filtered = df[np.array(predicted) < 0.3]
        df_filtered.to_csv('../data/twitter/process_smaller_train.csv', index=False)

        # Calculate accuracy
        accuracy = sum([a == b for a, b in zip(df_scores['Label'], predicted)]) / len(df_scores['Label'])
        print("Accuracy: ", accuracy)


if __name__ == '__main__':
    main()
