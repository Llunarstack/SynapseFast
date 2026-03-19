from __future__ import annotations

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=20_000)
    p.add_argument("--n_features", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import numpy as np

    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n_samples, args.n_features), dtype=np.float32)
    # Synthetic labels from a random linear separator
    w = rng.standard_normal((args.n_features,), dtype=np.float32)
    y = (X @ w > 0).astype(np.int64)

    n_train = int(0.8 * args.n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    import synapsefast as sf

    clf = sf.Pipeline(
        steps=[
            ("scaler", sf.StandardScaler()),
            ("knn", sf.KNNClassifier(k=7)),
        ]
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"synapsefast Pipeline(StandardScaler->KNN) accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
