from typing import Literal, Iterable

import torch
from mapping import Mapping
import pytest

from unittest import mock

from tests.utils.assert_sys import assert_no_out_arr
from tests.utils.random_sparse import randint_sparse


@assert_no_out_arr
def test_mapping_repeat_last_dims():
    tensor_test = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2, 2, 2, 3],
            [0, 0, 1, 0, 3, 1, 3, 3, 2],
            [2, 3, 1, 2, 3, 1, 2, 3, 1],
        ]
    )
    mapping_test = Mapping.repeat_last_dims(tensor_test, 2, 2)
    assert (
        mapping_test.target
        == torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3, 3, 2],
                [2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 3, 0, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 2],
                [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            ]
        )
    ).all()
    assert (
        mapping_test._batch
        == torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8],
                [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 5, 6, 7, 5, 6, 7, 8],
            ]
        )
    ).all()


@assert_no_out_arr
def test_mapping_selector():
    source = torch.randint(0, 16, (3, 8))
    target = torch.randint(0, 16, (5, 32))
    batch = torch.randint(0, 16, (3, 32))
    mapping = Mapping(source, target, batch)

    assert len(mapping) == 3

    with mock.patch("torch.Tensor.__getitem__") as mock_getitem:
        mapping[0].batch
        mock_getitem.assert_called_once_with(0)

    with mock.patch("torch.Tensor.__getitem__") as mock_getitem:
        mapping[1].batch
        mock_getitem.assert_called_once_with(1)

    with mock.patch("torch.Tensor.__getitem__") as mock_getitem:
        mapping[-1].batch
        mock_getitem.assert_called_once_with(-1)

    with pytest.raises(AssertionError):
        mapping[-3].batch

    with pytest.raises(AssertionError):
        mapping[3].batch


def sparse_to_dense(
    indices: torch.LongTensor, values: torch.Tensor, shape=Iterable[int]
) -> torch.Tensor:
    result = torch.zeros(tuple(shape), dtype=values.dtype)
    result[tuple(indices)] = values
    return result


def dense_to_sparse(dense: torch.Tensor) -> tuple[torch.LongTensor, torch.Tensor]:
    indices = dense.nonzero().t()
    values = dense[tuple(indices)]
    return indices, values


def assert_mapping_reduce_sum(
    mapping: Mapping,
    target: torch.Tensor,
    source: torch.Tensor,
):
    _, values = dense_to_sparse(target)
    result = mapping.reduce(values, "sum")
    _, expected_result = dense_to_sparse(source)
    assert (result == expected_result).all()


@assert_no_out_arr
def test_mapping_reduce():
    torch.manual_seed(0)

    indices, values = randint_sparse((16, 16), min_v=1, ratio=0.5)
    source = sparse_to_dense(indices, values, (16, 16))

    target = source[:, :, None] * source[:, None, :]
    result = target.sum(dim=2)

    mapping = Mapping.repeat_last_dims(indices, 1, 2)

    assert_mapping_reduce_sum(mapping, target, result)

    target = source[:, :, None, None] * source[None, None, :, :]
    result = target.sum(dim=(2, 3))

    mapping = Mapping.repeat_last_dims(indices, 2, 2)

    assert_mapping_reduce_sum(mapping, target, result)

    target = source[:, :, None] * source[:, None, :]
    result = target.sum(dim=2)

    mapping = Mapping.repeat_last_dims(indices, 1, 2)

    assert_mapping_reduce_sum(mapping[0], target, result)

    target = source[:, :, None] * source[:, None, :]
    result = target.sum(dim=1)

    mapping = Mapping.repeat_last_dims(indices, 1, 2)

    assert_mapping_reduce_sum(mapping[1], target, result)


@assert_no_out_arr
def test_mapping_broadcast():
    torch.manual_seed(0)

    batch = torch.randint(0, 64, (1024, 3))
    values = torch.randint(0, 2048, (64,))
    mapping = Mapping(
        torch.randint(0, 64, (1, 64)), torch.randint(0, 1024, (1, 1024)), batch
    )
    assert (mapping.broadcast(values) == values[batch[0]]).all()
    assert (mapping[0].broadcast(values) == values[batch[0]]).all()
    assert (mapping[1].broadcast(values) == values[batch[1]]).all()
    assert (mapping[2].broadcast(values) == values[batch[2]]).all()
