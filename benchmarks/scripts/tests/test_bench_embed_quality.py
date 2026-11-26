"""Tests for bench_embed_quality.py — embedding quality benchmark metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench_embed_quality import (
    ndcg_at_k,
    mean_average_precision,
    mrr_at_k,
    cosine_similarity_matrix,
    build_quality_summary,
    RETRIEVAL_DATASETS,
    STS_DATASETS,
    CLASSIFICATION_DATASETS,
    CLUSTERING_DATASETS,
)


def test_ndcg_perfect():
    assert ndcg_at_k([1.0, 0.0, 0.0], k=3) == 1.0


def test_ndcg_worst():
    assert ndcg_at_k([0.0, 0.0, 1.0], k=3) < 1.0
    assert ndcg_at_k([0.0, 0.0, 1.0], k=3) > 0.0


def test_ndcg_empty():
    assert ndcg_at_k([], k=10) == 0.0


def test_ndcg_all_zero():
    assert ndcg_at_k([0.0, 0.0, 0.0], k=3) == 0.0


def test_ndcg_graded_relevance():
    scores_good = ndcg_at_k([3.0, 2.0, 1.0], k=3)
    scores_bad = ndcg_at_k([1.0, 2.0, 3.0], k=3)
    assert scores_good > scores_bad


def test_ndcg_k_truncation():
    full = ndcg_at_k([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], k=10)
    partial = ndcg_at_k([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], k=1)
    assert full == partial == 1.0


def test_map_perfect():
    rels = [[1.0, 1.0, 1.0]]
    assert mean_average_precision(rels) == 1.0


def test_map_worst():
    rels = [[0.0, 0.0, 0.0]]
    assert mean_average_precision(rels) == 0.0


def test_map_mixed():
    rels = [[1.0, 0.0, 1.0]]
    result = mean_average_precision(rels)
    assert 0.0 < result < 1.0


def test_map_empty():
    assert mean_average_precision([]) == 0.0


def test_mrr_perfect():
    rels = [[1.0, 0.0, 0.0]]
    assert mrr_at_k(rels, k=10) == 1.0


def test_mrr_second():
    rels = [[0.0, 1.0, 0.0]]
    assert mrr_at_k(rels, k=10) == 0.5


def test_mrr_none():
    rels = [[0.0, 0.0, 0.0]]
    assert mrr_at_k(rels, k=10) == 0.0


def test_mrr_truncated():
    rels = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    assert mrr_at_k(rels, k=10) == 0.0
    assert mrr_at_k(rels, k=11) > 0.0


def test_cosine_similarity_identity():
    import numpy as np
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = cosine_similarity_matrix(a, a)
    assert abs(sim[0][0] - 1.0) < 1e-6
    assert abs(sim[1][1] - 1.0) < 1e-6
    assert abs(sim[0][1]) < 1e-6


def test_cosine_similarity_shape():
    import numpy as np
    a = np.array([[1.0, 0.0, 0.0]] * 5)
    b = np.array([[0.0, 1.0, 0.0]] * 3)
    sim = cosine_similarity_matrix(a, b)
    assert sim.shape == (5, 3)


def test_datasets_exist():
    assert len(RETRIEVAL_DATASETS) >= 2
    assert len(STS_DATASETS) >= 1
    assert len(CLASSIFICATION_DATASETS) >= 1
    assert len(CLUSTERING_DATASETS) >= 1


def test_retrieval_dataset_has_queries_and_corpus():
    for ds_name, ds in RETRIEVAL_DATASETS.items():
        assert len(ds["queries"]) >= 15
        assert len(ds["corpus"]) >= 15
        assert len(ds["queries"]) == len(ds["corpus"])


def test_sts_dataset_has_pairs():
    for ds_name, ds in STS_DATASETS.items():
        assert len(ds["pairs"]) >= 20
        for pair in ds["pairs"]:
            assert len(pair) == 3
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)
            assert isinstance(pair[2], (int, float))
            assert 0 <= pair[2] <= 5.0


def test_classification_dataset_has_samples():
    for ds_name, ds in CLASSIFICATION_DATASETS.items():
        assert len(ds["samples"]) >= 20
        labels = set(s[1] for s in ds["samples"])
        assert len(labels) >= 3


def test_clustering_dataset_has_items():
    for ds_name, ds in CLUSTERING_DATASETS.items():
        assert len(ds["items"]) >= 20
        topics = set(item[1] for item in ds["items"])
        assert len(topics) >= 3


def test_build_quality_summary_empty():
    summary = build_quality_summary({})
    assert summary == {}


def test_build_quality_summary_sts():
    report = {
        "sts": {"stsb": {"status": "ok", "spearman": 0.85}},
    }
    summary = build_quality_summary(report)
    assert summary["sts_spearman"] == 0.85


def test_build_quality_summary_classification():
    report = {
        "classification": {"intent": {"status": "ok", "accuracy": 0.75}},
    }
    summary = build_quality_summary(report)
    assert summary["classification_accuracy"] == 0.75


def test_build_quality_summary_clustering():
    report = {
        "clustering": {"topic_clusters": {"status": "ok", "v_measure": 0.65}},
    }
    summary = build_quality_summary(report)
    assert summary["clustering_v_measure"] == 0.65


def test_build_quality_summary_retrieval():
    report = {
        "retrieval": {
            "nfcorpus": {"status": "ok", "ndcg_at_10": 0.45},
            "scifact": {"status": "ok", "ndcg_at_10": 0.55},
        },
    }
    summary = build_quality_summary(report)
    assert 0.49 < summary["retrieval_ndcg_at_10"] < 0.51


def test_build_quality_summary_dimensions():
    report = {
        "dimension_reduction": {
            "768": {"summary": {"spearman": 0.87, "accuracy": 0.47}},
            "256": {"summary": {"spearman": 0.85, "accuracy": 0.45}},
        },
    }
    summary = build_quality_summary(report)
    assert "dimension_curve" in summary
    assert "768" in summary["dimension_curve"]
    assert "256" in summary["dimension_curve"]


def test_build_quality_summary_skipped_results():
    report = {
        "sts": {"stsb": {"status": "error", "reason": "timeout"}},
        "classification": {"intent": {"status": "skipped"}},
    }
    summary = build_quality_summary(report)
    assert "sts_spearman" not in summary
    assert "classification_accuracy" not in summary
