from __future__ import annotations

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    import synapsefast as sf

    clf = sf.HFTextClassifier(device=args.device)
    out = clf.predict(
        [
            "SynapseFast is building a fast ML stack.",
            "This is the worst thing I have ever seen.",
        ]
    )
    print(out)


if __name__ == "__main__":
    main()
