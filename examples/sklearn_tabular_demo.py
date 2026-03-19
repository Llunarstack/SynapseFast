from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50_000)
    parser.add_argument("--n_features", type=int, default=64)
    parser.add_argument("--n_informative", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        n_redundant=0,
        n_clusters_per_class=2,
        random_state=args.seed,
    )
    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=200, n_jobs=-1))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"sklearn LogisticRegression accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
