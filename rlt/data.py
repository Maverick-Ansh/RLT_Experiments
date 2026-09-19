"""Pregenerated, indexed, GPU-resident data streams.

App. J.4: "The three initialization seeds share the same indexed training stream and held-out
examples for each task."  So the stream is a function of (task, data-seed) only -- never of the
initialization seed -- and is cached on disk so every run of a task reads the identical bytes.
"""
import os
import numpy as np
import torch

from . import tasks as T

CACHE = '/content/RLT_Experiments/data'


def _seed_for(task, split):
    if task == 'addition':
        return {'train': T.ADD_SEED, 'val': T.ADD_SEED + 1, 'test': T.ADD_SEED + 2}[split]
    return {'train': T.TRAIN_SEED_FORMAL, 'val': T.VAL_SEED_FORMAL,
            'test': T.TEST_SEED_FORMAL}[split]


def _npz(task, split, n, length):
    tag = f'{task}_{split}_{n}_{length}'
    p = os.path.join(CACHE, tag + '.npz')
    if not os.path.exists(p):
        x, y, m = T.make(task, n, _seed_for(task, split), split=split, length=length)
        np.savez_compressed(p, x=x.astype(np.int16), y=y.astype(np.int16), m=m)
    d = np.load(p)
    return d['x'].astype(np.int64), d['y'].astype(np.int64), d['m']


def train_stream(task, n, device):
    """One contiguous stream of n examples; batch i is stream[i*B:(i+1)*B]."""
    x, y, m = _npz(task, 'train', n, None)
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
            torch.from_numpy(m).to(device))


def val_sets(task, n_per_len, device):
    """Sec 3.1 in-distribution validation: one set per configured length, pooled before averaging
    (App. J.3: "Their three equally sized validation groups are pooled within a run")."""
    out = []
    for L in T.VAL_LENGTHS[task]:
        x, y, m = _npz(task, 'val', n_per_len, L)
        out.append((L, torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
                    torch.from_numpy(m).to(device)))
    return out


def test_set(task, n, length, device):
    x, y, m = _npz(task, 'test', n, length)
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
            torch.from_numpy(m).to(device))
