r"""Helpers for affinity classification metric reporting.

Notation used in this module:

.. math::

    C_{ab} = \#\{i : y_i = a,\ \hat{y}_i = b\}

where rows are true labels and columns are predicted labels.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, multilabel_confusion_matrix, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels


def _resolve_labels(y_true, y_pred, labels: Sequence | None) -> list:
    if labels is not None:
        return list(labels)

    return list(unique_labels(y_true, y_pred))


def build_affinity_class_metrics(
    y_true,
    y_pred,
    labels: Sequence | None = None,
    class_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    r"""Build per-class affinity metrics for a multiclass classification task.

    For each class ``c``, the function reports:

    .. math::

        \mathrm{Precision}_c = \frac{C_{cc}}{\sum_{a=0}^{K-1} C_{ac}}

    .. math::

        \mathrm{Recall}_c = \frac{C_{cc}}{\sum_{b=0}^{K-1} C_{cb}}

    One-vs-rest accuracy is computed as:

    .. math::

        \mathrm{Acc}^{\mathrm{ovr}}_c = \frac{TP_c + TN_c}{N}

    with

    .. math::

        TP_c = C_{cc},\quad FN_c = \sum_{b \ne c} C_{cb},\quad
        FP_c = \sum_{a \ne c} C_{ac},\quad
        TN_c = \sum_{a \ne c} \sum_{b \ne c} C_{ab}.
    """

    label_order = _resolve_labels(y_true, y_pred, labels)
    display_names = list(class_names) if class_names is not None else [str(label) for label in label_order]

    if len(display_names) != len(label_order):
        raise ValueError("class_names must match the number of labels")

    precision = precision_score(y_true, y_pred, labels=label_order, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=label_order, average=None, zero_division=0)
    confusions = multilabel_confusion_matrix(y_true, y_pred, labels=label_order)

    true_negative = confusions[:, 0, 0]
    false_positive = confusions[:, 0, 1]
    false_negative = confusions[:, 1, 0]
    true_positive = confusions[:, 1, 1]
    support = true_positive + false_negative
    one_vs_rest_accuracy = (true_positive + true_negative) / (
        true_positive + true_negative + false_positive + false_negative
    )

    metrics_df = pd.DataFrame(
        {
            "Class": display_names,
            "Label": label_order,
            "Support": support.astype(int),
            "Precision": precision,
            "Recall": recall,
            "One_vs_Rest_Accuracy": one_vs_rest_accuracy,
            "True_Positive": true_positive.astype(int),
            "False_Positive": false_positive.astype(int),
            "False_Negative": false_negative.astype(int),
            "True_Negative": true_negative.astype(int),
        }
    )
    return metrics_df


def build_affinity_overall_metrics(y_true, y_pred) -> pd.DataFrame:
    r"""Build overall ordinal metrics for affinity prediction.

    The returned table includes:

    .. math::

        \mathrm{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}(\hat{y}_i = y_i)

    .. math::

        \kappa_w = 1 - \frac{\sum_{a=0}^{K-1}\sum_{b=0}^{K-1} w_{ab} C_{ab}}
        {\sum_{a=0}^{K-1}\sum_{b=0}^{K-1} w_{ab} E_{ab}}

    with quadratic weights

    .. math::

        w_{ab} = \frac{(a-b)^2}{(K-1)^2}

    and mean signed error

    .. math::

        \mathrm{MSE}_{\mathrm{signed}} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i).
    """

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    overall_metrics = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Quadratic Weighted Kappa",
                "Mean Signed Error",
            ],
            "Value": [
                float(np.mean(y_true_array == y_pred_array)),
                float(cohen_kappa_score(y_true_array, y_pred_array, weights="quadratic")),
                float(np.mean(y_pred_array - y_true_array)),
            ],
        }
    )
    return overall_metrics


def aggregate_one_vs_rest_accuracy(
    y_true,
    y_pred,
    labels: Sequence | None = None,
    class_names: Sequence[str] | None = None,
) -> float:
    r"""Aggregate classwise one-vs-rest accuracy into a single group-level score.

    The group score is the support-weighted mean of the classwise one-vs-rest
    accuracies:

    .. math::

    \mathrm{Acc}_{\mathrm{group}} = \frac{\sum_c s_c\,\mathrm{Acc}^{\mathrm{ovr}}_c}
    {\sum_c s_c}

    where ``s_c`` is the support of class ``c`` in the group.
    """

    class_metrics_df = build_affinity_class_metrics(
        y_true,
        y_pred,
        labels=labels,
        class_names=class_names,
    )
    weights = class_metrics_df["Support"].to_numpy(dtype=float)
    scores = class_metrics_df["One_vs_Rest_Accuracy"].to_numpy(dtype=float)

    if weights.sum() == 0:
        return float(np.mean(scores))

    return float(np.average(scores, weights=weights))