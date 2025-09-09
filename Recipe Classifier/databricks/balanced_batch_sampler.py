import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader, BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        """
        labels - a list of unique ids per datapoint, e.g. path
        n_classes - number of classes in a batch
        n_samples - number of samples in each class
        """
        self.labels = labels
        # self.labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0]
            for label in self.labels_set
        }

        for l in self.labels_set:
            np.random.shuffle(
                self.label_to_indices[l]
            )  # To make sure different samples are selected each time
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = min(
            n_classes, len(self.labels_set)
        )  # Make sure n_classes is not greater than available classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][
                        self.used_label_indices_count[
                            class_
                        ] : self.used_label_indices_count[class_]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[class_] += self.n_samples

                if self.used_label_indices_count[class_] + self.n_samples > len(
                    self.label_to_indices[class_]
                ):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            # yield indices
            # self.count += self.n_classes + self.n_samples
            # self.count += self.batch_size <- GPT

            # Only yield if indices have the expected batch size
            if len(indices) == self.batch_size:
                yield indices
                self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size
        # return math.ceil(self.n_dataset / self.batch_size)
