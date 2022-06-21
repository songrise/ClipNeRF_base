# -*- coding : utf-8 -*-
# @FileName  : rand_batch.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Jun 19, 2022
# @Github    : https://github.com/songrise
# @Description: Random batch sampler
#%%
import torch
class RandBatchSampler(torch.utils.data.Sampler):
    """
    Random batch sampler.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_samples = len(data_source)
        self.n_batches = self.num_samples // self.batch_size

    def __iter__(self):
        batch_sequence = torch.randperm(self.n_batches)
        for i in range(self.n_batches):
            batch =[]
            for j in range(self.batch_size):
                batch.append(batch_sequence[i]*self.batch_size+j)
            yield batch

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class BatchSampler(torch.utils.data.Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        

    def __iter__(self) :
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
#%%   
if __name__ == '__main__':
    s = BatchSampler(torch.arange(0,20), 4, False)
    s = RandBatchSampler(torch.arange(0,100),10)
    for _ in range(1):
        for i in s:
            print(i)
# %%
