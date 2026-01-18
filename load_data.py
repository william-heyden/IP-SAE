#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import io
from sae_helper import NormalizeFea

# Default: use ./data next to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def set_data_root(data_root: str) -> None:
    """
    Set the dataset root folder.
    Expected structure:
        <data_root>/AWA2/res101.mat
        <data_root>/AWA2/att_splits.mat
        <data_root>/CUB/res101.mat
        <data_root>/CUB/att_splits.mat
        <data_root>/SUN/res101.mat
        <data_root>/SUN/att_splits.mat
        <data_root>/AWA1/res101.mat
        <data_root>/AWA1/att_splits.mat
    """
    global DATA_DIR
    DATA_DIR = os.path.abspath(data_root)


def _load_xlsa17(dataset_name: str):
    ds_dir = os.path.join(DATA_DIR, dataset_name)
    res101_path = os.path.join(ds_dir, "res101.mat")
    att_splits_path = os.path.join(ds_dir, "att_splits.mat")

    if not os.path.exists(res101_path):
        raise FileNotFoundError(f"Missing: {res101_path}")
    if not os.path.exists(att_splits_path):
        raise FileNotFoundError(f"Missing: {att_splits_path}")

    res101 = io.loadmat(res101_path)
    att_splits = io.loadmat(att_splits_path)
    return res101, att_splits


def _split_and_pack(
    res101,
    att_splits,
    train_loc: str,
    val_loc: str,
    test_loc: str,
    norm_data: bool,
    int_proj: bool,
    paired_sample: bool = True,
    feature_keep_idx=None,  # used for CUB attribute filtering
):
    """
    Shared logic for all XLSA17 datasets.
    Optionally supports CUB feature/attribute filtering.
    """

    labels = res101["labels"]
    labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc] - 1)])
    labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc] - 1)])
    labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc] - 1)])

    train_labels_seen = np.unique(labels_train)
    val_labels_unseen = np.unique(labels_val)
    test_labels_unseen = np.unique(labels_test)

    # Convert class ids to 0-based
    train_labels_seen = np.array([x - 1 for x in train_labels_seen])
    val_labels_unseen = np.array([x - 1 for x in val_labels_unseen])
    test_labels_unseen = np.array([x - 1 for x in test_labels_unseen])

    # Remap split labels to consecutive ids (original style)
    i = 0
    for lab in train_labels_seen:
        labels_train[labels_train == lab] = i
        i += 1
    j = 0
    for lab in val_labels_unseen:
        labels_val[labels_val == lab] = j
        j += 1
    k = 0
    for lab in test_labels_unseen:
        labels_test[labels_test == lab] = k
        k += 1

    sig = att_splits["att"]  # [A x C]

    # Optional attribute filtering (CUB original code)
    if feature_keep_idx is not None:
        sig = sig[feature_keep_idx, :]

    test_sig = sig[:, test_labels_unseen]  # unseen prototypes

    testClasses = test_labels_unseen
    trainClasses = np.concatenate([train_labels_seen, val_labels_unseen], axis=0)

    # Features / labels
    X = res101["features"].transpose()  # [N x D]
    Y_temp = res101["labels"].transpose()[0]
    Y = np.array([x - 1 for x in Y_temp]).astype(np.int32)

    # Class attribute table
    att = att_splits["att"].transpose()  # [C x A]

    # Optional attribute filtering (CUB original code)
    if feature_keep_idx is not None:
        att = att[:, feature_keep_idx]  # [C x A_filtered]

    noExs = X.shape[0]

    trainDataX, trainDataLabels, trainDataAttrs = [], [], []
    testDataX, testDataLabels, testDataAttrs = [], [], []

    for ii in range(noExs):
        if Y[ii] in trainClasses:
            trainDataX.append(X[ii])
            trainDataLabels.append(Y[ii])
            trainDataAttrs.append(att[Y[ii]])
        elif Y[ii] in testClasses:
            testDataX.append(X[ii])
            testDataLabels.append(Y[ii])
            testDataAttrs.append(att[Y[ii]])
        else:
            raise RuntimeError("Fatal Error... Please check code/data")

    X_tr = np.array(trainDataX)
    X_te = np.array(testDataX)
    S_tr = np.array(trainDataAttrs)

    Y_tr = np.array(trainDataLabels)
    Y_te = np.array(testDataLabels)
    S_te_all = np.array(testDataAttrs)

    if norm_data:
        X_tr = NormalizeFea(X_tr.T, 2).T

    if int_proj:
        X_tr = np.concatenate([X_tr, S_tr], axis=1)

    S_te_pro = np.transpose(test_sig)  # [C_unseen x A]
    te_cl_id = np.unique(Y_te)

    if int_proj:
        if paired_sample:
            X_te = np.concatenate([X_te, S_te_all], axis=1)
        else:
            X_te = np.concatenate([X_te, S_te_all[np.random.permutation(S_te_all.shape[0])]], axis=1)

    if norm_data:
        S_te_pro = NormalizeFea(S_te_pro.T, 2).T

    return X_tr, X_te, S_tr, S_te_pro, Y_te, te_cl_id, Y_tr, S_te_all, att


# -----------------------------
# Dataset loaders
# -----------------------------

def awa2(norm_data: bool = True, int_proj: bool = False, paired_sample: bool = True):
    res101, att_splits = _load_xlsa17("AWA2")
    return _split_and_pack(
        res101=res101,
        att_splits=att_splits,
        train_loc="train_loc",
        val_loc="val_loc",
        test_loc="test_unseen_loc",
        norm_data=norm_data,
        int_proj=int_proj,
        paired_sample=paired_sample,
        feature_keep_idx=None
    )


def awa1(norm_data: bool = True, int_proj: bool = False, paired_sample: bool = True):
    res101, att_splits = _load_xlsa17("AWA1")
    return _split_and_pack(
        res101=res101,
        att_splits=att_splits,
        train_loc="train_loc",
        val_loc="val_loc",
        test_loc="test_unseen_loc",
        norm_data=norm_data,
        int_proj=int_proj,
        paired_sample=paired_sample,
        feature_keep_idx=None
    )


def sun(norm_data: bool = True, int_proj: bool = False, paired_sample: bool = True):
    res101, att_splits = _load_xlsa17("SUN")
    return _split_and_pack(
        res101=res101,
        att_splits=att_splits,
        train_loc="train_loc",
        val_loc="val_loc",
        test_loc="test_unseen_loc",
        norm_data=norm_data,
        int_proj=int_proj,
        paired_sample=paired_sample,
        feature_keep_idx=None
    )


def cub(norm_data: bool = True, int_proj: bool = False, paired_sample: bool = True):
    """
    CUB uses a subset of attributes/features in the original codebase.
    Your old file removed indices 'a' from the attribute dimension.
    Here we replicate that behavior by KEEPING the complement.
    """
    res101, att_splits = _load_xlsa17("CUB")

    # Indices removed in the original file (1-based in comment, but used as python indices)
    # Your previous code: np.delete(sig, a, axis=0)
    # so 'a' are indices to delete (0-based).
    a_delete = [
        1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
        51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99,
        101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151,
        152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193,
        194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235,
        236, 238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274,
        277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311
    ]

    # Determine full attribute dimension from att_splits['att'] which is [A x C]
    A = att_splits["att"].shape[0]
    keep = np.array([i for i in range(A) if i not in set(a_delete)], dtype=int)

    return _split_and_pack(
        res101=res101,
        att_splits=att_splits,
        train_loc="train_loc",
        val_loc="val_loc",
        test_loc="test_unseen_loc",
        norm_data=norm_data,
        int_proj=int_proj,
        paired_sample=paired_sample,
        feature_keep_idx=keep
    )
