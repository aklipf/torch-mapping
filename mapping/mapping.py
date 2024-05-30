from __future__ import annotations
from typing import Tuple, Any, Literal

import torch
import torch.nn.functional as F


class Mapping:
    def __init__(
        self,
        source: torch.LongTensor,
        target: torch.LongTensor,
        batch: torch.LongTensor,
        _selected: int = 0,
    ):
        assert isinstance(source, torch.LongTensor)
        assert isinstance(target, torch.LongTensor)
        assert batch.ndim == 2 and batch.dtype == torch.long
        assert -batch.shape[0] < _selected < batch.shape[0]

        self._batch = batch
        self._source = source
        self._target = target
        self._selected = _selected

    def __len__(self) -> int:
        return self._batch.shape[0]

    def __getitem__(self, idx: int) -> Mapping.Selector:
        return Mapping(self.source, self.target, self._batch, idx)

    def reduce(
        self,
        tensor: torch.Tensor,
        reduce: Literal["sum", "mean"],
    ) -> torch.Tensor:
        assert tensor.shape[0] == self._target.shape[1]

        batch = self.batch
        while batch.ndim < tensor.ndim:
            batch.unsqueeze_(-1)

        reduced = torch.zeros(
            (self._source.shape[1], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        ).scatter_reduce_(
            dim=0, index=batch.expand_as(tensor), src=tensor, reduce=reduce
        )

        return reduced

    def broadcast(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        assert tensor.shape[0] == self._source.shape[1]

        return tensor[self.batch]

    @classmethod
    def repeat_last_dims(
        cls, indices: torch.LongTensor, ndim: int = 1, repeat: int = 2
    ) -> Mapping:
        boadcasted_indices, mapping = cls._repeat_last_dims(indices, ndim, repeat)

        return cls(source=indices, target=boadcasted_indices, batch=mapping)

    @classmethod
    def _repeat_last_dims(
        cls, indices: torch.LongTensor, ndim: int, repeat: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        assert ndim > 0 and repeat > 1
        assert ndim <= indices.shape[0]

        # calculate base and reprtition
        unique_indices, count = Mapping._count_repeating_indices(indices[:-ndim])
        count_repeat = count.pow(repeat)

        repeated_base = unique_indices.repeat_interleave(count_repeat, dim=1)

        # calculating indexing of the repeated dims
        ptr_base = F.pad(count.cumsum(0), (1, 0))
        ptr_top = F.pad(count_repeat.cumsum(0), (1, 0))
        batch_top = torch.arange(
            count_repeat.shape[0], dtype=torch.long, device=indices.device
        ).repeat_interleave(count_repeat)
        idx_top = (
            torch.arange(batch_top.shape[0], dtype=torch.long, device=indices.device)
            - ptr_top[batch_top]
        )

        exp_top = F.pad(
            count[None, :].repeat(repeat - 1, 1).cumprod(0), (0, 0, 1, 0), value=1
        ).flip(0)

        idx_top = (
            idx_top[None, :] // exp_top[:, batch_top] % count[None, batch_top]
            + ptr_base[None, batch_top]
        )

        # repeat top dims and concatenate with base
        repeated_top = (
            indices[-ndim:, idx_top].swapdims(0, 1).reshape(ndim * repeat, -1)
        )
        result_indices = torch.cat((repeated_base, repeated_top), dim=0)

        return result_indices, idx_top

    @staticmethod
    def _count_repeating_indices(
        indices: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        change = F.pad(
            (indices[:, :-1] != indices[:, 1:]).any(dim=0), (1, 0), value=True
        )
        _, count = torch.unique_consecutive(change.cumsum(0), return_counts=True)
        unique_indices = indices[:, change]

        return unique_indices, count

    @property
    def source(self) -> torch.LongTensor:
        return self._source

    @property
    def target(self) -> torch.LongTensor:
        return self._target

    @property
    def batch(self) -> torch.LongTensor:
        return self._batch[self._selected]
