import argparse
from bin import __main__ as nlu_benchmark
from rasa_data_to_csv import rasa_data_to_csv
import os


def main(threshold: str, nlu_data_dir: str):
    print(f"Threshold: {threshold}")
    print(f"NLU data dir: {nlu_data_dir}")
    print("Running NLU benchmark")

    if os.path.isdir(nlu_data_dir): 
        filepath = os.path.join(nlu_data_dir, "test.yml")
    else:
        filepath = nlu_data_dir
        nlu_data_dir = os.path.dirname(filepath)
    ftype = nlu_data_dir.split(".")[-1] # get file type from nlu_data_dir
    nlu_benchmark.check(threshold=threshold, nlu_data_dir=nlu_data_dir, ftype=ftype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLU benchmark")
    parser.add_argument("--threshold", type=str, help="Threshold for NLU benchmark")
    parser.add_argument("--nlu-data-dir", type=str, help="NLU data directory")
    args = parser.parse_args()
    main(args.threshold, args.nlu_data_dir)