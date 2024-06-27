import torch
from os import path as osp
import logging
import numpy as np
from torch.utils.data import Subset, DataLoader

from utils.logger import get_root_logger
from utils.misc import get_time_str
from data.dataset import DancerDataset
from model.model_pipeline import Pipeline

def init_logger():
    log_file = osp.join("log", f"train_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='ai_choreo', log_level=logging.INFO, log_file=log_file)

    return logger


def create_dataset_loader():
    dancer_np = np.load('data/processed_dyads_rehearsal_leah.npy')
    dataset = DancerDataset(torch.from_numpy(dancer_np))

    train_size = int(0.8 * len(dataset))

    train_dataset = Subset(dataset, range(train_size))
    test_dataset = Subset(dataset, range(train_size, len(dataset)))

    print(len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger = init_logger()
    train_loader, test_loader = create_dataset_loader()

    epochs = 10

    logger.info("Started training.")

    training_losses = []

    model = Pipeline()

    for epoch in range(epochs):
        for train_data in train_loader:
            model.feed_data(train_data)
            model.optimize_parameters()
            training_losses.append(model.loss)
            print('loss:', model.loss)

        model.update_learning_rate()


if __name__ == '__main__':
    main()
