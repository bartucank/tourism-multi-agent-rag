import argparse

from rag import build_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["heading", "character"], default="heading")
    args = parser.parse_args()
    build_index(args.method)
