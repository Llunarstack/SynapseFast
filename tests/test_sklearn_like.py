import numpy as np

import synapsefast as sf


def test_standard_scaler_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 8)).astype(np.float32) * 3.0 + 10.0
    sc = sf.StandardScaler()
    Xt = sc.fit_transform(X)
    m = Xt.mean(axis=0)
    v = Xt.var(axis=0)
    assert np.allclose(m, 0.0, atol=1e-2)
    assert np.allclose(v, 1.0, atol=1e-2)


def test_pipeline_fit_predict_runs():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 16), dtype=np.float32)
    w = rng.standard_normal((16,), dtype=np.float32)
    y = (X @ w > 0).astype(np.int64)
    n = 1500
    Xtr, Xte = X[:n], X[n:]
    ytr, yte = y[:n], y[n:]

    clf = sf.Pipeline([("scaler", sf.StandardScaler()), ("knn", sf.KNNClassifier(k=5))])
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    assert pred.shape == yte.shape
    acc = sf.accuracy(yte, pred)
    assert 0.5 <= acc <= 1.0
