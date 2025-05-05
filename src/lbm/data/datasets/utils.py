import logging
from typing import List, Union

import numpy as np
from torch.utils.data import IterableDataset
from webdataset import DataPipeline


class RandomSampleMultiDatasets(IterableDataset):
    """
    Randomly sample from multiple datasets with given probabilities.

    Args:

        datasets (List[Union[IterableDataset, DataPipeline]]): list of datasets to sample from
        probabilities (List[float]): list of probabilities for each dataset. If None, the datasets will be sampled uniformly.
            Defaults to None
    """

    def __init__(
        self,
        datasets: List[Union[IterableDataset, DataPipeline]],
        probabilities: List[float] = None,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.probabilities = probabilities
        self.n_datasets = len(datasets)
        assert self.n_datasets > 0, "datasets should not be empty"

        if probabilities is not None:
            assert len(datasets) == len(
                probabilities
            ), "datasets and probabilities should have the same length"
            assert all(
                0 <= p <= 1 for p in probabilities
            ), "probabilities should be between 0 and 1"
            assert sum(probabilities) == 1, "probabilities should sum to 1"

        else:
            # set uniform sampling
            probabilities = [1 / self.n_datasets] * self.n_datasets

        self.probabilities = probabilities

    def __iter__(self):
        while True:
            try:
                if self.n_datasets == 1:
                    yield from self.datasets[0]
                else:
                    dataset_id = np.random.choice(self.n_datasets, p=self.probabilities)
                    logging.debug(f"Sampling from dataset {dataset_id}")
                    dataset = self.datasets[dataset_id]
                    yield next(iter(dataset))
            except StopIteration:
                return
