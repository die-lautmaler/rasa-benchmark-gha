import os
import typer
import sys
from dotenv import load_dotenv
from typing import Optional, List
from googleapiclient.errors import HttpError

from .benchmark_storage import BenchmarkStorage
from .benchmark import Benchmark, BenchmarkResult, BenchmarkRunner, BenchmarkType
from .benchmark_report import BenchmarkReport
from .benchmark_gsheet import BenchmarkGsheet

load_dotenv()

benchmarker = typer.Typer()


FULL_SET_NAME = "all_sets"
NLU_DATA_DIR = "../../data/"
SCORE_THRESHOLD = 0.8


@benchmarker.command()
def full_run(
    testset_names: Optional[List[str]] = typer.Argument(
        None,
        help="one or several testset names to create test report. Only the newest testrun will be loaded",
    ),
    run_id: Optional[str] = typer.Option(
        default="nlu_update",
        help="if run_id was used in benchmark execution, load results for run_id (only newest as well)",
    ),
    data_root: Optional[str] = typer.Option(
        None, help="path to folder holding the test set data"
    ),
    ftype: Optional[str] = typer.Option(
        default="csv", help="file type csv|json (atm only csv is implemented)"
    ),
    testtype: Optional[str] = "nlu_r",
    zip_path: Optional[str] = typer.Option(
        None, help="if zipping is wanted, you must pass the path to report-folder"
    ),
):
    typer.echo("running benchmark and report creation")
    if not data_root:
        typer.echo("using default data root")
        os.chdir(os.path.split(__file__)[0])
        data_root = NLU_DATA_DIR

    run(testset_names, ftype, data_root, run_id, testtype)
    report(testset_names, data_root, run_id, ftype, zip_path)
    typer.echo("full run Done")


@benchmarker.command()
def run(
    testset_names: Optional[List[str]] = None,
    ftype: Optional[str] = "csv",
    data_root: Optional[str] = None,
    run_id: Optional[str] = "nlu_update",
    testtype: Optional[str] = "nlu_r",
):
    """
        run a benchmark of given test sets against a running nlu service
    :param testset_names:
    :param ftype:
    :param data_root:
    :param run_id:
    :param testtype:
    :return:
    """
    if not data_root:
        typer.echo("using default data root")
        os.chdir(os.path.split(__file__)[0])
        data_root = NLU_DATA_DIR

    storage = BenchmarkStorage(data_root, ftype, run_id, "de")

    if not testset_names:
        testset_names = storage.get_testset_names()
        typer.echo(
            "no testset name given running all sets {}".format("\n".join(testset_names))
        )
    else:
        typer.echo("tsets {}".format(testset_names))

    all_results = BenchmarkResult(run_id, FULL_SET_NAME)
    # if BenchmarkType[testtype.upper()] == BenchmarkType.NLU_D:
    #     typer.echo(f'Agent under Test has a total of {len(load_intents())} intents')
    testset_names.sort()
    for testset in testset_names:
        typer.secho("\nrunning testset {}".format(testset), fg=typer.colors.BLUE)
        try:
            tasks = storage.load_test_set(testset)
            benchmark = Benchmark(
                run_id, testset, BenchmarkType[testtype.upper()], tasks
            )
            benchmark_runner = BenchmarkRunner(benchmark, 2)
            benchmark_runner.start()
            result = benchmark_runner.result
            storage.save_results(result)

            score = result.get_match_rate()
            n_tests = score[0]
            matchscore = score[3]

            typer.echo(f"Number of Tests: {n_tests}, Matchscore {matchscore}")
            all_results.add_results(result.get_results())

        except Exception as e:
            # we did not get a valid testfile
            typer.secho(e.__str__(), err=True)

    storage.save_results(all_results)
    return (n_tests, matchscore)


def check(
    run_id: Optional[str] = "nlu_update",
    ftype: Optional[str] = "csv",
    testtype: Optional[str] = "nlu_r",
    nlu_data_dir: Optional[str] = "nlu.csv",
    threshold: Optional[str] = typer.Option(
        default=SCORE_THRESHOLD, help="threshold for score"
    ),
    ):
    """
    check if trained model reaches score > threshold
    """
    typer.echo(f"check if trained model reaches score > {threshold}")
    n_tests, score = run(data_root=nlu_data_dir, run_id=run_id, ftype=ftype, testtype=testtype)
    
    if score < float(threshold):
        typer.secho(f"score {score} is below threshold", fg=typer.colors.RED)
        sys.exit(1)
    else:
        typer.secho(f"score {score} is above threshold", fg=typer.colors.GREEN)
        sys.exit(0)


@benchmarker.command()
def report(
    testset_names: Optional[List[str]] = typer.Argument(
        None,
        help="one or several testset names to create test report. Only the newest testrun will be loaded",
    ),
    data_root: Optional[str] = typer.Option(
        None, help="path to folder holding the test set data"
    ),
    run_id: Optional[str] = typer.Option(
        default="nlu_update",
        help="if run_id was used in benchmark execution, load results for run_id (only newest as well)",
    ),
    ftype: Optional[str] = typer.Option(
        default="csv", help="file type csv|json (atm only csv is implemented)"
    ),
    zip_path: Optional[str] = typer.Option(
        None, help="if zipping is wanted, you must pass the path to report-folder"
    ),
):
    """
    create a report on a set of test results\n
    - confusion matrix plot
    - confidence histogram plot
    - per intent report with performance measure precision, recall, f1
    - plus overall scores
    """
    if not data_root:
        typer.echo("using default data root")
        os.chdir(os.path.split(__file__)[0])
        data_root = "../data"

    storage = BenchmarkStorage(data_root, ftype, run_id, "de")
    storage.results_housekeeping()

    if not testset_names:
        typer.secho("no testset name given running available results")
        testset_names = storage.get_resultset_names()
    typer.secho("creating reports for:\n{}\n".format("\n".join(testset_names)))

    do_full_set = False
    if FULL_SET_NAME in testset_names:
        do_full_set = True
        testset_names = [x for x in testset_names if x != FULL_SET_NAME]

    testset_names.sort()
    for testset_name in testset_names:
        typer.secho(
            "\ncreating report for {}".format(testset_name), fg=typer.colors.BLUE
        )
        benchmark_result = storage.load_result_df(testset_name)
        bench_report = BenchmarkReport(benchmark_result, testset_name, run_id)
        bench_report.create_report(
            os.path.join(data_root, "results/reports/", testset_name)
        )

    if do_full_set:
        typer.secho(
            "\ncreating report for {}".format(FULL_SET_NAME), fg=typer.colors.BLUE
        )
        prev_report = storage.load_prev_fullset_report()
        prev_results = storage.load_result_df(FULL_SET_NAME, True)
        intent_map = storage.load_intent_map()
        benchmark_result = storage.load_result_df(FULL_SET_NAME)
        bench_report = BenchmarkReport(benchmark_result, FULL_SET_NAME, run_id)
        bench_report.create_fullset_report(
            os.path.join(data_root, "results/reports/", FULL_SET_NAME),
            prev_report,
            prev_results,
            intent_map,
        )

    if zip_path:
        storage.store_report_zip(zip_path)

    return


@benchmarker.command()
def gsheet(
    target: Optional[str] = typer.Argument(
        "appweb", help="set sales to upload to sales nlu reports"
    ),
    data_root: Optional[str] = typer.Option(
        None, help="path to folder holding the test set data"
    ),
    run_id: Optional[str] = typer.Option(
        default="nlu_update",
        help="if run_id was used in benchmark execution, load results for run_id (only newest as well)",
    ),
):
    if not data_root:
        typer.echo("using default data root")
        os.chdir(os.path.split(__file__)[0])
        data_root = NLU_DATA_DIR

    folder_id = "1lms7zyo3lj1YPFVuxXwJS1SC5xpfkh06"
    if target == "sales":
        folder_id = "1mbNHN7uIKYPz1t1NAU52ULR3s9PxkKIy"

    storage = BenchmarkStorage(data_root, "csv", run_id, "de", FULL_SET_NAME)

    try:
        gsheet = BenchmarkGsheet(folder_id, run_id)
        gsheet.nlu_update_report(
            storage.load_result_df(FULL_SET_NAME),
            storage.load_reports(),
            storage.load_switch_df(),
            storage.load_conf_matrix_df(),
        )
        gsheet.move_spreadsheet()
        typer.secho("Done", fg=typer.colors.GREEN)
    except HttpError as error:
        typer.echo(f"An error occurred: {error}", err=True)
        return error


@benchmarker.command()
def optfb(
    target: Optional[str] = typer.Argument(
        "appweb", help="set sales to upload to sales nlu reports"
    ),
    data_root: Optional[str] = typer.Option(
        None, help="path to folder holding the test set data"
    ),
    run_id: Optional[str] = typer.Option(
        default="nlu_update",
        help="if run_id was used in benchmark execution, load results for run_id (only newest as well)",
    ),
):
    if not data_root:
        typer.echo("using default data root")
        os.chdir(os.path.split(__file__)[0])
        data_root = NLU_DATA_DIR

    storage = BenchmarkStorage(data_root, "csv", run_id, "de", FULL_SET_NAME)
    all_results = storage.load_result_df(FULL_SET_NAME)
    fbth_df = BenchmarkReport.compute_optimal_fb_thresh(all_results, 0.05)

    typer.echo("saving result to {}".format(os.path.join(data_root, "fbth.csv")))
    fbth_df.to_csv(os.path.join(data_root, "fbth.csv"))


if __name__ == "__main__":
    benchmarker()
