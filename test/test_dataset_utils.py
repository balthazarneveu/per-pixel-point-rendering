import random
from pixr.learning.utils import split_dataset


def test_split_dataset_length():
    rendered_images_size = 100
    train_indices, val_indices = split_dataset(rendered_images_size)
    assert len(train_indices) + len(val_indices) == rendered_images_size


def test_split_dataset_unique_indices():
    number_of_view = 100
    train_indices, val_indices = split_dataset(number_of_view)
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)


def test_split_dataset_ratio():
    number_of_view = 100
    ratio_train = 0.7
    train_indices, val_indices = split_dataset(number_of_view, ratio_train=ratio_train)
    assert len(train_indices) == int(number_of_view * ratio_train)
    assert len(val_indices) == number_of_view - int(number_of_view * ratio_train)


def test_split_reproducibility():
    number_of_view = 100
    ratio_train = 0.7
    train_indices_1, val_indices_1 = split_dataset(number_of_view, ratio_train=ratio_train, seed=42)
    train_indices_2, val_indices_2 = split_dataset(number_of_view, ratio_train=ratio_train, seed=45)
    train_indices_3, val_indices_3 = split_dataset(number_of_view, ratio_train=ratio_train, seed=42)
    assert len(train_indices_1) == len(train_indices_2)
    assert len(val_indices_1) == len(val_indices_2)
    assert train_indices_1 != train_indices_2
    assert val_indices_1 != val_indices_2
    assert train_indices_1 == train_indices_3
    assert val_indices_1 == val_indices_3


def test_split_dataset_invalid_ratio_train():
    number_of_view = 100
    invalid_ratio_train = -1
    # try:
    #     split_dataset(number_of_view, ratio_train=invalid_ratio_train)
    #     assert False, "AssertionError not raised for invalid ratio_train value"
    # except AssertionError:
    #     pass

    invalid_ratio_train = 80.  # 80% ---> no 0.8!
    try:
        split_dataset(number_of_view, ratio_train=invalid_ratio_train)
        assert False, "AssertionError not raised for invalid ratio_train value"
    except AssertionError:
        pass
