from typing import Tuple

import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, Subset

Indexes = Tuple[np.ndarray, np.ndarray, np.ndarray]
Datasets = Tuple[Dataset, Dataset, Dataset]


class PhotoScreenDataset(ImageFolder):
    def __init__(self, root, **kwargs):
        super(PhotoScreenDataset, self).__init__(root, **kwargs)


    def _split_train_valid_test(self, train_rate: float, valid_rate: float=None,
                                test_rate: float=None, shuffle: bool=False) -> Indexes:
        """
        Create indexes for 3 basics datasets splitting
        :param train_rate: train ratio
        :param valid_rate: validation ratio
        :param test_rate: test ratio
        :return: three numpy arrays with indexes
        """
        if valid_rate is None:
            valid_rate = 0.
        elif test_rate is None:
            test_rate = 0.
        
        assert train_rate + valid_rate + test_rate <= 1.0

        train_size = int(train_rate * len(self))
        valid_size = int(valid_rate * len(self))

        all_indexes = np.arange(len(self))
        train_indexes = np.random.choice(all_indexes, size=train_size, replace=False)
        all_indexes = np.setdiff1d(all_indexes, train_indexes)
        valid_indexes = np.random.choice(all_indexes, size=valid_size, replace=False)
        test_indexes = np.setdiff1d(all_indexes, valid_indexes)

        if shuffle:
            np.random.shuffle(train_indexes)
            np.random.shuffle(valid_indexes)
            np.random.shuffle(test_indexes)

        return train_indexes, valid_indexes, test_indexes
    

    def split_dataset(self, train_rate: float, valid_rate: float=None, 
                    test_rate: float=None, shuffle: bool=False) -> Datasets:
        train_indexes, valid_indexes, test_indexes = self._split_train_valid_test(
            train_rate, valid_rate, test_rate, shuffle
        )
        train_subset = Subset(self, train_indexes)
        valid_subset = Subset(self, valid_indexes)
        test_subset = Subset(self, test_indexes)
        
        return train_subset, valid_subset, test_subset

