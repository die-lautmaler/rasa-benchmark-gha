import argparse
import sys
import typer

from bin import __main__ as nlu_benchmark
import os


def main(threshold: str, test_data_source: str):
    typer.secho(f"Threshold: {threshold}")
    typer.secho(f"NLU data dir: {test_data_source}")
    typer.secho("Running NLU benchmark")

    if not os.path.exists(test_data_source):
        raise FileNotFoundError(f"File or Directory {test_data_source} does not exist. Cannot proceed")

    test_set_names = None
    if os.path.isdir(test_data_source):
        path_segments = os.path.split(test_data_source.rstrip(os.path.sep))
        test_root = path_segments[0]
        ftype = path_segments[1]
    else:
        path, fname = os.path.split(test_data_source)
        test_root, ftype = os.path.split(path)
        test_set_names = [fname.split(".")[-2]]

    n_tests, correct, incorrect, score = nlu_benchmark.run(testset_names=test_set_names, data_root=test_root,
                                                           run_id='nlu_test',
                                                           ftype=ftype)

    if score < float(threshold):
        typer.secho(f"score {score} is below threshold {threshold}", fg=typer.colors.RED)
        sys.exit(1)
    else:
        typer.secho(f"score {score} is above threshold {threshold}", fg=typer.colors.GREEN)
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLU benchmark")
    parser.add_argument("--threshold", type=str, help="Threshold for NLU benchmark")
    parser.add_argument("--nlu-data", type=str, help="NLU data directory")
    args = parser.parse_args()
    main(args.threshold, args.nlu_data)
