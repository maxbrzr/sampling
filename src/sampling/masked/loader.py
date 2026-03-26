from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from metalearning_benchmarks.metalearning_benchmark import MetaLearningBenchmark  # type: ignore
from torch import Tensor


class MetaBenchmarkDataset(Dataset[Tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, benchmark: MetaLearningBenchmark, context_size: int):
        """
        Args:
            benchmark: The initialized LineSine1D benchmark object.
            context_size: Number of points to use for the context set (few-shot).
        """
        self.benchmark = benchmark
        self.context_size = context_size

    def __len__(self) -> int:
        # The benchmark usually defines n_task
        return int(self.benchmark.n_task)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # 1. Retrieve task data
        task = self.benchmark.get_task_by_index(idx)

        # Ensure data is (N, 1). .view(-1, 1) guarantees the feature dim exists.
        # Original code used .squeeze(1) which might have flattened it to (N,)
        x_all = torch.from_numpy(task.x).float().view(-1, 1)
        y_all = torch.from_numpy(task.y).float().view(-1, 1)

        n_points = x_all.shape[0]

        # 2. Randomly split into Context and Target
        # Permute indices to select random context points
        perm = torch.randperm(n_points)

        idx_ctx = perm[: self.context_size]
        idx_tar = perm  # usually we evaluate on ALL points (reconstruction) or just the held-out ones

        x_ctx, y_ctx = x_all[idx_ctx], y_all[idx_ctx]
        x_tar, y_tar = x_all[idx_tar], y_all[idx_tar]
        # Shapes: (N_ctx, 1), (N_ctx, 1), (N_tar, 1), (N_tar, 1)

        return x_ctx, y_ctx, x_tar, y_tar


def get_meta_loader(
    benchmark: MetaLearningBenchmark, batch_size: int, context_size: int
) -> DataLoader[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    dataset = MetaBenchmarkDataset(benchmark, context_size=context_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
