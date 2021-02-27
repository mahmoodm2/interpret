# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..postprocessing import multiclass_postprocess
import numpy as np


def test_multiclass_postprocess_smoke():
    n = 1000
    d = 2
    k = 3
    b = 10

    X_binned = np.random.randint(b, size=(d, n))
    feature_graphs = []
    for _ in range(d):
        feature_graphs.append(np.random.rand(b, k))

    def binned_predict_proba(X_binned, k=3):
        n = X_binned.shape[1]
        return 1 / k * np.ones((n, k))

    feature_types = ["numeric"] * d
    results = multiclass_postprocess(
        X_binned, feature_graphs, binned_predict_proba, feature_types
    )

    assert "intercepts" in results
    assert "feature_graphs" in results

# TODO merging multiple ebms

    # data = adult_classification()
    # X = data["full"]["X"]
    # y = data["full"]["y"]
    # X_tr = data["train"]["X"]
    # y_tr = data["train"]["y"]
    # X_te = data["test"]["X"]
    # y_te = data["test"]["y"]



#   valid_ebm(clf)

#     global_exp = clf.explain_global()
#     local_exp = clf.explain_local(X_te[:5, :], y_te[:5])

#     _smoke_test_explanations(global_exp, local_exp, 6000)