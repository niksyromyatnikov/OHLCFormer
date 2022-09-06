import copy
import random
import torch
from ohlcformer.losses import Loss, MaskedDirectionLoss
from tests import generate_configs_combinations


def test_loss_init(loss_cls, configs_combinations: list, default_configs: dict) -> bool:
    res = []

    for configs in configs_combinations:
        loss = loss_cls(**configs)
        cfgs = copy.copy(default_configs)
        cfgs.update({k: v for k, v in configs.items() if v is not None})

        for key, value in cfgs.items():
            res.append(getattr(loss, key) == value)

    return all(res)


def default_test_loss_init(loss_cls=None) -> bool:
    loss_cls = loss_cls if loss_cls is not None else Loss
    configs = {'weighted': [True, False], 'reduction': ['sum', 'mean'], 'name': [None, 'test_loss']}
    default_configs = {'weighted': False, 'reduction': 'sum', 'name': loss_cls.__name__}
    configs_combinations = list(generate_configs_combinations(**configs))

    return test_loss_init(loss_cls, configs_combinations, default_configs)


def test_loss_forward(loss_cls, torch_loss_cls, test_f1: bool = False) -> bool:
    l1 = loss_cls(reduction='sum')
    l2 = loss_cls(reduction='mean')
    criterion = torch_loss_cls()
    epsilon = 1e-03

    shape = (8, 50, 4)

    for _ in range(100):
        x = torch.randn(shape)
        mask = torch.empty(shape[:-1]).random_(2)
        y = torch.randn(shape)
        f1 = 0.0
        tp, tn, fp, fn = 0, 0, 0, 0
        return_f1 = random.choice([0, 1])
        ignored_index = random.choices(population=[0, 1], weights=[0.95, 0.05])[0]

        params = {'prediction': x, 'target': y, 'mask': mask, 'ignored_index': ignored_index}

        if test_f1:
            params.update({'return_f1': return_f1})

        l1_res = l1(**params)
        l2_res = l2(**params)

        if test_f1 and return_f1:
            l1_res, f1 = l1_res
            l2_res, f1 = l2_res

        output = []
        for row_idx, row in enumerate(x):
            idx_y = 0
            for elem_idx, elem in enumerate(row):
                if mask[row_idx][elem_idx] != ignored_index:
                    elem_x = elem
                    elem_y = y[row_idx][idx_y]

                    if loss_cls == MaskedDirectionLoss:
                        elem_x = torch.signbit(elem_x)
                        elem_y = torch.signbit(elem_y)

                        for i, x in enumerate(elem_x):
                            tp += x == 1 and elem_y[i] == 1
                            tn += x == 0 and elem_y[i] == 0
                            fp += x == 1 and elem_y[i] == 0
                            fn += x == 0 and elem_y[i] == 1

                    err = criterion(elem_x.float(), elem_y.float())
                    output.append(err)
                    idx_y += 1

        if sum(output) * shape[-1] - l1_res.item() > epsilon:
            raise AssertionError(f'Expected {sum(output) * shape[-1]} but got {l1_res.item()}')

        if len(output) > 0 and sum(output) / len(output) - l2_res.item() > epsilon or l2_res.item() == 0:
            raise AssertionError(f'Expected {sum(output) / len(output)} but got {l2_res.item()}')

        if test_f1 and return_f1:
            div = (2 * tp + fp + fn)
            if div != 0 and f1 != 2 * tp / div:
                raise AssertionError(f'Expected {2 * tp / (2 * tp + fp + fn)} but got {f1}')

    return True
