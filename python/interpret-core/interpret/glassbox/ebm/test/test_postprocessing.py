# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..postprocessing import multiclass_postprocess
from ..ebm import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from .test_ebm import valid_ebm, _smoke_test_explanations
from interpret.glassbox.ebm.utils import *
from ....test.utils import  adult_classification
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
    
def test_merge_models():
    
    data = adult_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]   
    
    seed =1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    ebm1 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)

    ebm1.fit(X_train, y_train)  

    seed +=10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    ebm2 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
    ebm2.fit(X_train, y_train)  

    seed +=10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    ebm3 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
    ebm3.fit(X_train, y_train) 
        
    models = [ebm1, ebm2 , ebm3]
    merged_ebm = EBMUtils.merge_models(models=models)

    ebm_global = merged_ebm.explain_global(name='EBM')
    
    valid_ebm(merged_ebm)

    global_exp = merged_ebm.explain_global()
    local_exp = merged_ebm.explain_local(X_te[:5, :], y_te[:5])

    _smoke_test_explanations(global_exp, local_exp, 6000) 

