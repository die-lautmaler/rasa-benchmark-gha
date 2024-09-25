import argparse
from bin import __main__ as nlu_benchmark


def main(threshold: str, nlu_data_dir: str):
    print(f"Threshold: {threshold}")
    print(f"NLU data dir: {nlu_data_dir}")
    print("Running NLU benchmark")
    nlu_benchmark.check(threshold=threshold, nlu_data_dir=nlu_data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLU benchmark")
    parser.add_argument("--threshold", type=str, help="Threshold for NLU benchmark")
    parser.add_argument("--nlu_data_dir", type=str, help="NLU data directory")
    args = parser.parse_args()
    main(args.threshold, args.nlu_data_dir)