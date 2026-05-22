#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import pickle
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import torch
from transformers import AutoModel
from datasets import load_from_disk

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from geneformer import TranscriptomeTokenizer


# =========================
# User settings
# =========================

PROJECT_ROOT = Path("/Users/zhuqin/Desktop/CMML/ICA2")

RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)

RUN_PREPARE_INPUTS = True
RUN_TOKENIZE = True
RUN_EXTRACT_EMBEDDINGS = True
RUN_EVALUATE = True

# If True, regenerate files even if they already exist.
OVERWRITE_INPUTS = False
OVERWRITE_TOKENIZED = False
OVERWRITE_EMBEDDINGS = False

# CPU-friendly settings
DEVICE = torch.device("cpu")
BATCH_SIZE = 4
MAX_LEN = 2048

CLASSES = ["alpha", "beta", "delta", "pp"]

# Processed count h5ad files
PROCESSED_FILES = {
    "ref": PROJECT_ROOT / "data/processed/pancreas_ref_counts_endocrine4_sharedgenes.h5ad",
    "control": PROJECT_ROOT / "data/processed/pancreas_query_control_counts_endocrine4_sharedgenes.h5ad",
    "aab": PROJECT_ROOT / "data/processed/pancreas_query_aab_counts_endocrine4_sharedgenes.h5ad",
    "t1d": PROJECT_ROOT / "data/processed/pancreas_query_t1d_counts_endocrine4_sharedgenes.h5ad",
}

# Output directories
GENEFORMER_INPUT_DIR = PROJECT_ROOT / "data/processed/geneformer_input"
TOKEN_DIR = PROJECT_ROOT / "data/processed/geneformer_tokenized_v1"
RESULT_DIR = PROJECT_ROOT / "results/geneformer_v1_pilot"
EMB_DIR = RESULT_DIR / "embeddings"

# Geneformer repo downloaded with Git LFS
GENEFORMER_ROOT = PROJECT_ROOT / "models/geneformer_hf"

# GC30M Geneformer V1 dictionaries
GENE_MEDIAN_FILE = (
    GENEFORMER_ROOT
    / "geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl"
)
TOKEN_DICTIONARY_FILE = (
    GENEFORMER_ROOT
    / "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl"
)
GENE_MAPPING_FILE = (
    GENEFORMER_ROOT
    / "geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl"
)

# Balanced pilot sampling design
REF_N_PER_CLASS = {
    "alpha": 100,
    "beta": 100,
    "delta": 100,
    "pp": 121,
}

QUERY_N_PER_CLASS = {
    "alpha": 50,
    "beta": 50,
    "delta": 50,
    "pp": 50,
}


# =========================
# Helper functions
# =========================

def ensure_dirs():
    GENEFORMER_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)


def check_pickle_file(path: Path):
    print(f"\nChecking dictionary file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print("size:", path.stat().st_size)
    with open(path, "rb") as f:
        head = f.read(100)

    print("head:", repr(head[:40]))
    if head.startswith(b"version https://"):
        raise RuntimeError(
            f"{path} is still a Git LFS pointer, not the real pickle file.\n"
            "Run git lfs pull inside models/geneformer_hf."
        )

    with open(path, "rb") as f:
        obj = pickle.load(f)

    print(f"Loaded OK: type={type(obj)}, length={len(obj)}")
    return obj


def check_geneformer_dictionaries():
    print("\n========== Checking Geneformer V1 dictionaries ==========")
    check_pickle_file(GENE_MEDIAN_FILE)
    check_pickle_file(TOKEN_DICTIONARY_FILE)
    check_pickle_file(GENE_MAPPING_FILE)


def find_label_col(adata):
    candidates = [
        "cell_type",
        "celltype",
        "label",
        "cell_type_harmonized",
        "harmonized_label",
        "endocrine_label",
        "celltype_harmonized",
    ]
    for c in candidates:
        if c in adata.obs.columns:
            return c

    raise ValueError(
        "Cannot find label column. Available obs columns:\n"
        + ", ".join(adata.obs.columns.astype(str))
    )


def harmonise_labels(values):
    """
    Convert possible original pancreas labels to alpha/beta/delta/pp.
    If already harmonised, keep as is.
    """
    label_map = {
        "pancreatic A cell": "alpha",
        "type B pancreatic cell": "beta",
        "pancreatic D cell": "delta",
        "pancreatic PP cell": "pp",
        "PP cell": "pp",
        "alpha": "alpha",
        "beta": "beta",
        "delta": "delta",
        "pp": "pp",
    }

    values = pd.Series(values).astype(str)
    mapped = values.map(label_map)

    if mapped.isna().any():
        missing = sorted(values[mapped.isna()].unique())
        raise ValueError(f"Unmapped labels found: {missing}")

    return mapped.values


def find_ensembl_ids(adata):
    """
    Return Ensembl IDs from var_names or common var columns.
    """
    vnames = pd.Index(adata.var_names.astype(str))
    vnames_no_version = vnames.str.replace(r"\.\d+$", "", regex=True)

    if vnames_no_version.str.startswith("ENSG").mean() > 0.5:
        print("Using adata.var_names as Ensembl IDs")
        return pd.Series(vnames_no_version.values, index=adata.var_names)

    candidates = [
        "ensembl_id",
        "feature_id",
        "gene_id",
        "gene_ids",
        "id",
    ]

    for c in candidates:
        if c in adata.var.columns:
            vals = adata.var[c].astype(str).str.replace(r"\.\d+$", "", regex=True)
            if vals.str.startswith("ENSG").mean() > 0.5:
                print(f"Using adata.var['{c}'] as Ensembl IDs")
                return pd.Series(vals.values, index=adata.var_names)

    raise ValueError(
        "No Ensembl IDs found. Check adata.var_names and adata.var columns:\n"
        + ", ".join(adata.var.columns.astype(str))
    )


def sample_balanced(adata, label_col, n_per_class):
    selected = []

    for lab, n in n_per_class.items():
        idx = np.where(adata.obs[label_col].astype(str).values == lab)[0]

        if len(idx) == 0:
            raise ValueError(f"No cells found for label: {lab}")

        take = min(n, len(idx))
        chosen = np.random.choice(idx, size=take, replace=False)
        selected.extend(chosen)

        print(
            f"{lab}: requested {n}, available {len(idx)}, selected {take}"
        )

    selected = np.array(selected)
    np.random.shuffle(selected)

    return adata[selected].copy()


def prepare_geneformer_inputs():
    print("\n========== Preparing Geneformer pilot h5ad inputs ==========")

    for name, path in PROCESSED_FILES.items():
        out_path = GENEFORMER_INPUT_DIR / f"pancreas_{name}_geneformer_pilot.h5ad"

        if out_path.exists() and not OVERWRITE_INPUTS:
            print(f"\nSkipping existing input: {out_path}")
            continue

        print(f"\nProcessing {name}")
        print("Input:", path)

        if not path.exists():
            raise FileNotFoundError(path)

        adata = sc.read_h5ad(path)
        adata.obs_names_make_unique()
        adata.var_names_make_unique()

        label_col = find_label_col(adata)
        adata.obs["cell_type"] = harmonise_labels(adata.obs[label_col].values)

        # Keep only target endocrine classes
        keep_cells = adata.obs["cell_type"].isin(CLASSES).values
        adata = adata[keep_cells].copy()

        if name == "ref":
            n_per_class = REF_N_PER_CLASS
        else:
            n_per_class = QUERY_N_PER_CLASS

        adata = sample_balanced(adata, "cell_type", n_per_class)

        ens = find_ensembl_ids(adata)
        ens = ens.astype(str).str.replace(r"\.\d+$", "", regex=True)

        keep_genes = (
            ens.notna()
            & ens.str.startswith("ENSG")
            & ~ens.duplicated()
        )

        print(f"Keeping {keep_genes.sum()} / {adata.n_vars} genes with unique Ensembl IDs")

        adata = adata[:, keep_genes.values].copy()
        ens = ens[keep_genes]

        adata.var["ensembl_id"] = ens.values
        adata.var["filter_pass"] = 1

        if sparse.issparse(adata.X):
            n_counts = np.asarray(adata.X.sum(axis=1)).ravel()
        else:
            n_counts = np.asarray(adata.X.sum(axis=1)).ravel()

        adata.obs["n_counts"] = n_counts
        adata.obs["filter_pass"] = 1
        adata.obs["cohort"] = name

        adata.write_h5ad(out_path)

        print(f"Saved: {out_path}")
        print("Shape:", adata.shape)
        print(adata.obs["cell_type"].value_counts())


def tokenize_geneformer_inputs():
    print("\n========== Tokenizing Geneformer inputs ==========")

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={
            "cell_type": "cell_type",
            "cohort": "cohort",
        },
        nproc=1,
        model_version="V1",
        model_input_size=2048,
        special_token=False,
        collapse_gene_ids=False,
        gene_median_file=str(GENE_MEDIAN_FILE),
        token_dictionary_file=str(TOKEN_DICTIONARY_FILE),
        gene_mapping_file=str(GENE_MAPPING_FILE),
    )

    for name in ["ref", "control", "aab", "t1d"]:
        in_file = GENEFORMER_INPUT_DIR / f"pancreas_{name}_geneformer_pilot.h5ad"
        out_dataset = TOKEN_DIR / f"pancreas_{name}_geneformer_v1_pilot.dataset"

        if out_dataset.exists() and not OVERWRITE_TOKENIZED:
            print(f"\nSkipping existing tokenized dataset: {out_dataset}")
            continue

        print(f"\nTokenizing {name}")
        print("Input:", in_file)

        if not in_file.exists():
            raise FileNotFoundError(in_file)

        tmp_dir = GENEFORMER_INPUT_DIR / f"tmp_geneformer_v1_{name}"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        tmp_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(in_file, tmp_dir / in_file.name)

        output_prefix = f"pancreas_{name}_geneformer_v1_pilot"

        tk.tokenize_data(
            data_directory=str(tmp_dir),
            output_directory=str(TOKEN_DIR),
            output_prefix=output_prefix,
            file_format="h5ad",
        )

        print(f"Finished tokenizing {name}")

        shutil.rmtree(tmp_dir)


def inspect_tokenized_datasets():
    print("\n========== Inspecting tokenized datasets ==========")

    for name in ["ref", "control", "aab", "t1d"]:
        ds_path = TOKEN_DIR / f"pancreas_{name}_geneformer_v1_pilot.dataset"

        if not ds_path.exists():
            raise FileNotFoundError(ds_path)

        ds = load_from_disk(str(ds_path))

        print(f"\n{name}: {ds_path}")
        print(ds)
        print("columns:", ds.column_names)

        if len(ds) > 0:
            print("first keys:", ds[0].keys())

            if "cell_type" in ds.column_names:
                print("cell_type example:", ds[0]["cell_type"])

            if "input_ids" in ds.column_names:
                print("input length example:", len(ds[0]["input_ids"]))


def find_geneformer_v1_model_dir():
    """
    Find local Geneformer V1-10M model directory.
    Prefer directories containing V1 or 10M in their path.
    """
    print("\n========== Searching for Geneformer V1 model checkpoint ==========")

    if not GENEFORMER_ROOT.exists():
        raise FileNotFoundError(
            f"{GENEFORMER_ROOT} does not exist. "
            "Clone the Geneformer repo with Git LFS first."
        )

    candidates = []

    for config_path in GENEFORMER_ROOT.rglob("config.json"):
        model_dir = config_path.parent

        has_weights = (
            (model_dir / "pytorch_model.bin").exists()
            or (model_dir / "model.safetensors").exists()
        )

        if has_weights:
            candidates.append(model_dir)

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No Geneformer model checkpoint found. Expected a directory containing "
            "config.json and pytorch_model.bin or model.safetensors.\n\n"
            "Try running:\n"
            "cd /Users/zhuqin/Desktop/CMML/ICA2/models/geneformer_hf\n"
            "git lfs pull --include=\"Geneformer-V1-10M/*,geneformer/gene_dictionaries_30m/*\" --exclude=\"\"\n"
        )

    print("Candidate model directories:")
    for c in candidates:
        print(" -", c)

    preferred = [
        c for c in candidates
        if "V1" in str(c) or "10M" in str(c) or "30M" in str(c)
    ]

    if len(preferred) > 0:
        model_dir = preferred[0]
    else:
        model_dir = candidates[0]

    print("\nSelected model directory:", model_dir)

    # Check whether weight file is still LFS pointer
    for weight_name in ["pytorch_model.bin", "model.safetensors"]:
        weight_path = model_dir / weight_name
        if weight_path.exists():
            with open(weight_path, "rb") as f:
                head = f.read(100)

            print("Weight file:", weight_path)
            print("size:", weight_path.stat().st_size)

            if head.startswith(b"version https://"):
                raise RuntimeError(
                    f"{weight_path} is a Git LFS pointer, not real model weights.\n"
                    "Run git lfs pull for the V1 model."
                )

    return model_dir


def pad_batch(input_ids_batch, max_len=2048, pad_token_id=0):
    """
    Pad and truncate a batch of input_ids.
    """
    batch_size = len(input_ids_batch)
    lengths = [min(len(x), max_len) for x in input_ids_batch]
    cur_max_len = max(lengths)

    input_ids = torch.full(
        (batch_size, cur_max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (batch_size, cur_max_len),
        dtype=torch.long,
    )

    for i, ids in enumerate(input_ids_batch):
        ids = ids[:cur_max_len]
        ids = torch.tensor(ids, dtype=torch.long)

        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1

    return input_ids, attention_mask


def mean_pool_hidden(last_hidden_state, attention_mask):
    """
    Mean-pool token embeddings using attention mask.
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def extract_embeddings_for_dataset(model, name):
    ds_path = TOKEN_DIR / f"pancreas_{name}_geneformer_v1_pilot.dataset"
    x_out = EMB_DIR / f"pancreas_{name}_geneformer_v1_X.npy"
    meta_out = EMB_DIR / f"pancreas_{name}_geneformer_v1_meta.csv"

    if x_out.exists() and meta_out.exists() and not OVERWRITE_EMBEDDINGS:
        print(f"\nSkipping existing embeddings for {name}")
        return

    print(f"\nExtracting embeddings for {name}")
    print("Dataset:", ds_path)

    ds = load_from_disk(str(ds_path))

    if "input_ids" not in ds.column_names:
        raise ValueError(f"No input_ids column found in {ds_path}")

    if "cell_type" not in ds.column_names:
        raise ValueError(f"No cell_type column found in {ds_path}")

    model.eval()

    all_embeddings = []
    all_labels = []
    all_cohorts = []

    n = len(ds)

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)

            batch = ds[start:end]
            input_ids_batch = batch["input_ids"]

            input_ids, attention_mask = pad_batch(
                input_ids_batch,
                max_len=MAX_LEN,
                pad_token_id=0,
            )

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            emb = mean_pool_hidden(
                outputs.last_hidden_state,
                attention_mask,
            )

            all_embeddings.append(emb.cpu().numpy())

            all_labels.extend(batch["cell_type"])

            if "cohort" in batch:
                all_cohorts.extend(batch["cohort"])
            else:
                all_cohorts.extend([name] * (end - start))

            if start % (BATCH_SIZE * 10) == 0:
                print(f"  processed {end}/{n}")

    X = np.vstack(all_embeddings)

    meta = pd.DataFrame(
        {
            "cell_type": all_labels,
            "cohort": all_cohorts,
        }
    )

    np.save(x_out, X)
    meta.to_csv(meta_out, index=False)

    print("Saved embeddings:", x_out, X.shape)
    print("Saved metadata:", meta_out)
    print(meta["cell_type"].value_counts())


def extract_all_embeddings():
    print("\n========== Extracting Geneformer V1 embeddings ==========")

    model_dir = find_geneformer_v1_model_dir()

    print("\nLoading model from:", model_dir)
    model = AutoModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded.")
    print("Device:", DEVICE)

    for name in ["ref", "control", "aab", "t1d"]:
        extract_embeddings_for_dataset(model, name)


def load_embedding_and_labels(name):
    x_path = EMB_DIR / f"pancreas_{name}_geneformer_v1_X.npy"
    meta_path = EMB_DIR / f"pancreas_{name}_geneformer_v1_meta.csv"

    if not x_path.exists():
        raise FileNotFoundError(x_path)

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    X = np.load(x_path)
    meta = pd.read_csv(meta_path)

    y = meta["cell_type"].astype(str).values

    return X, y, meta


def evaluate_geneformer_lr():
    print("\n========== Evaluating Geneformer V1 embeddings with Logistic Regression ==========")

    X_ref, y_ref, _ = load_embedding_and_labels("ref")

    print("Reference embeddings:", X_ref.shape)
    print(pd.Series(y_ref).value_counts())

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
    )

    clf.fit(X_ref, y_ref)

    summary_rows = []

    for name in ["control", "aab", "t1d"]:
        X, y, meta = load_embedding_and_labels(name)

        pred = clf.predict(X)

        acc = accuracy_score(y, pred)

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y,
            pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )

        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y,
            pred,
            labels=CLASSES,
            average="weighted",
            zero_division=0,
        )

        row = {
            "dataset": name,
            "n_cells": len(y),
            "accuracy": acc,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_p,
            "weighted_recall": weighted_r,
            "weighted_f1": weighted_f1,
        }

        summary_rows.append(row)

        print(f"\n===== {name} =====")
        print(pd.DataFrame([row]).to_string(index=False))

        # Per-class report
        report = classification_report(
            y,
            pred,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).T
        report_path = RESULT_DIR / f"geneformer_v1_{name}_classification_report.csv"
        report_df.to_csv(report_path)

        print("Saved classification report:", report_path)

        # Confusion matrix
        cm = confusion_matrix(y, pred, labels=CLASSES)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in CLASSES],
            columns=[f"pred_{c}" for c in CLASSES],
        )

        cm_path = RESULT_DIR / f"geneformer_v1_{name}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)

        print("Saved confusion matrix:", cm_path)

    summary = pd.DataFrame(summary_rows)
    summary_path = RESULT_DIR / "geneformer_v1_pilot_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved overall summary:")
    print(summary_path)
    print(summary.to_string(index=False))


def main():
    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)

    ensure_dirs()

    check_geneformer_dictionaries()

    if RUN_PREPARE_INPUTS:
        prepare_geneformer_inputs()

    if RUN_TOKENIZE:
        tokenize_geneformer_inputs()
        inspect_tokenized_datasets()

    if RUN_EXTRACT_EMBEDDINGS:
        extract_all_embeddings()

    if RUN_EVALUATE:
        evaluate_geneformer_lr()

    print("\nGeneformer V1 pilot pipeline completed.")


if __name__ == "__main__":
    main()
