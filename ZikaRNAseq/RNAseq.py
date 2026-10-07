"""Helpers for the ZikaRNAseq notebook (Python 3 port of the 2016 RNAseq.py)."""
import json
from time import sleep

import numpy as np
import requests
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA

from plots import COLORS10, enlarge_tick_fontsize

ENRICHR_URL = 'https://maayanlab.cloud/Enrichr'
CDS2_URL = 'https://maayanlab.cloud/L1000CDS2'


def cpm(counts):
    """Counts per million, column-wise (edgeR::cpm without normalization factors)."""
    return counts / counts.sum(axis=0) * 1e6


def perform_PCA(matrix, standardize=2, log=True):
    """PCA of samples (columns). Returns (percent variance of PC1-3, sample coordinates)."""
    if log:
        matrix = np.log10(matrix + 1.)
    if standardize == 2:  # z-score each gene across samples
        matrix = zscore(matrix, axis=1)
    elif standardize == 1:  # z-score each sample across genes
        matrix = zscore(matrix, axis=0)
    matrix = matrix[~np.isnan(matrix.sum(axis=1))]

    pca = PCA(n_components=None)
    pca.fit(matrix.T)
    variance_explained = pca.explained_variance_ratio_[0:3] * 100
    return variance_explained, pca.transform(matrix.T)


def PCA_plot(matrix, samples, standardize=2, log=True, legend_loc='best'):
    """2D PCA scatter of samples, colored by the group labels in `samples`."""
    variance_explained, transformed = perform_PCA(matrix, standardize=standardize, log=log)
    samples = np.asarray(samples)
    groups = list(dict.fromkeys(samples))

    fig, ax = plt.subplots(figsize=(8, 8))
    for group, color in zip(groups, COLORS10):
        mask = samples == group
        ax.scatter(transformed[mask, 0], transformed[mask, 1], color=color, s=50, label=group)
    ax.legend(frameon=True, loc=legend_loc, prop={'size': 14})
    ax.set_xlabel('PC1 (%.2f%% variance captured)' % variance_explained[0], fontsize=16)
    ax.set_ylabel('PC2 (%.2f%% variance captured)' % variance_explained[1], fontsize=16)
    enlarge_tick_fontsize(ax, 14)
    fig.tight_layout()
    plt.show()


def chdir(data, sampleclass, gamma=.5):
    """Characteristic Direction (Clark et al. 2014, BMC Bioinformatics 15:79).

    data: genes x samples array; sampleclass: 1 = control, 2 = treated, 0 = ignored.
    Returns one coefficient per gene; the vector has unit length.
    """
    sampleclass = np.asarray(sampleclass)
    keep = sampleclass != 0
    with np.errstate(invalid='ignore', divide='ignore'):  # constant genes become NaN and are dropped below
        data = zscore(np.asarray(data, dtype=np.float64)[:, keep], axis=1, ddof=1)
    sampleclass = sampleclass[keep]
    valid = ~np.isnan(data).any(axis=1)
    data = data[valid]

    mean_diff = data[:, sampleclass == 2].mean(axis=1) - data[:, sampleclass == 1].mean(axis=1)

    # Shrinkage LDA in the PCA space that keeps 99.9% of the variance.
    pca = PCA(n_components=None).fit(data.T)
    n_pc = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.999) + 1
    V = pca.components_[:n_pc].T

    # Within-class scatter: remove each class mean, then project onto the kept PCs.
    centered = data.copy()
    for c in (1, 2):
        centered[:, sampleclass == c] -= centered[:, sampleclass == c].mean(axis=1, keepdims=True)
    R = centered.T @ V
    within = (R.T @ R) / (data.shape[1] - 2)
    shrunk_cov = gamma * within + (1 - gamma) * np.eye(n_pc) * np.trace(within) / n_pc

    b = V @ np.linalg.solve(shrunk_cov, V.T @ mean_diff)
    b /= np.linalg.norm(b)

    coefs = np.zeros(valid.shape[0])  # genes with no variance get a zero coefficient
    coefs[valid] = b
    return coefs


def _enrichr_add_list(genes, meta=''):
    payload = {'list': (None, '\n'.join(genes)), 'description': (None, meta)}
    response = requests.post('%s/addList' % ENRICHR_URL, files=payload)
    response.raise_for_status()
    return response.json()


def enrichr_link(genes, meta=''):
    """POST a gene list to Enrichr and return the results page URL."""
    short_id = _enrichr_add_list(genes, meta)['shortId']
    return '%s/enrich?dataset=%s' % (ENRICHR_URL, short_id)


def enrichr_result(genes, meta='', gmt=''):
    """POST a gene list to Enrichr and return enrichment results for one gene-set library."""
    list_id = _enrichr_add_list(genes, meta)['userListId']
    sleep(2)
    response = requests.get('%s/enrich' % ENRICHR_URL,
                            params={'userListId': list_id, 'backgroundType': gmt})
    response.raise_for_status()
    return response.json()


def post_to_cds2(genes, vals, name=None, aggravate=False):
    """POST a CD signature to L1000CDS2 and return the full response (includes shareId)."""
    payload = {
        'data': {'genes': [g.upper() for g in genes], 'vals': list(vals)},
        'config': {'aggravate': aggravate, 'searchMethod': 'CD', 'share': True,
                   'combination': True, 'db-version': 'latest'},
        'meta': [{'key': 'name', 'value': name}],
    }
    response = requests.post('%s/query' % CDS2_URL, data=json.dumps(payload),
                             headers={'content-type': 'application/json'})
    response.raise_for_status()
    return response.json()
