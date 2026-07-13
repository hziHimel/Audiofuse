"""Tests for make_clean_split.py — the leakage-free split builder."""
import pandas as pd
from make_clean_split import load_master


def test_master_is_unique_and_expected_size():
    m = load_master()
    assert m["filename"].is_unique, "duplicate recordings in master list"
    assert len(m) == 3240, f"expected 3240 unique recordings, got {len(m)}"


def test_labels_are_binary_0_1():
    m = load_master()
    assert set(m["label"].unique()) <= {0, 1}


def test_label_mapping_matches_reference():
    # a0001 is abnormal (REFERENCE.csv: a0001,1) -> label 1
    m = load_master()
    assert m.loc[m["filename"] == "a0001", "label"].iloc[0] == 1


def test_class_balance_reasonable():
    m = load_master()
    frac = m["label"].mean()
    assert 0.18 < frac < 0.23, f"abnormal fraction {frac:.3f} outside expected ~0.205"


def test_clean_split_no_overlap():
    # regression test for the leakage bug: a clean split must have zero overlap
    from sklearn.model_selection import train_test_split
    m = load_master()
    strat = m["subset"] + "_" + m["label"].astype(str)
    tr, vl = train_test_split(m, test_size=0.2, random_state=42, stratify=strat)
    assert len(set(tr["filename"]) & set(vl["filename"])) == 0
    assert len(set(tr["npy_filepath"]) & set(vl["npy_filepath"])) == 0
