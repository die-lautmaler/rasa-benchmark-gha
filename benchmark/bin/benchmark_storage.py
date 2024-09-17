import csv
import pandas
import os
import shutil
import typer

from datetime import datetime
from typing import List, Union

from .benchmark import BenchmarkResult, Task


class BenchmarkStorage:
    """
    provides methods to load and save test sets, results and reports
    serves as abstraction layer to integrater database backend
    """

    data_root: str
    ftype: str
    run_id: str
    tset_dir: str
    language_code: str
    fullset_name: str

    def __init__(
        self,
        data_root: str,
        ftype: str,
        run_id: str,
        language_code: str = "de",
        fullset_name: str = "all_sets",
    ):
        if not (os.path.exists(data_root) or os.path.isdir(data_root)):
            raise ValueError(
                "data root {} does not exist or is not a directory".format(data_root)
            )
        if ftype not in ["csv", "json", "yaml"]:
            raise ValueError("file type {} can not be processed".format(ftype))

        self.data_root = data_root
        self.ftype = ftype
        self.tset_dir = os.path.join(data_root, ftype)
        self.run_id = run_id
        self.language_code = language_code
        self.fullset_name = fullset_name
        if not (os.path.exists(self.tset_dir) or os.path.isdir(self.tset_dir)):
            raise ValueError(
                "test sets root {} does not exist or is not a directory".format(
                    self.tset_dir
                )
            )

    def load_test_set(self, testset_name: str) -> List[Task]:
        tasks: List[Task] = []
        unlabeled = 0
        testset_path = os.path.join(
            self.tset_dir, "{}.{}".format(testset_name, self.ftype)
        )
        if not os.path.isfile(testset_path):
            typer.secho(
                "Please provide a a valid testset name", color=True, fg=typer.colors.RED
            )
            raise ValueError("no valid testset file found")
        else:
            with open(testset_path, "r") as tsetf:
                if self.ftype == "csv":
                    for line in csv.reader(tsetf):
                        if line[1]:
                            tasks.append(Task(line[0], line[1], self.language_code))
                        else:
                            unlabeled += 1
                elif self.ftype == "yaml":
                    raise TypeError("yaml files not supported yet")

        typer.echo("{} tasks loaded".format(len(tasks)))
        typer.echo("{} tasks without intent annotation not loaded".format(unlabeled))
        return tasks

    def load_result_df(
        self, testset_name: str, previous: bool = False
    ) -> pandas.DataFrame:
        """
            load benchmark results as pandas Dataframe
        :return:
        """

        result_file_path = self.get_result_file_path(testset_name, previous)
        if result_file_path:
            typer.echo("loading results file {}".format(result_file_path))
            with open(result_file_path, "r", encoding="utf8") as resf:
                if self.ftype == "csv":
                    return pandas.read_csv(resf, sep=",", decimal=",")
                else:
                    typer.echo(
                        "reading format {} not supported yet".format(self.ftype),
                        err=True,
                    )
                    return pandas.DataFrame()
        else:
            typer.echo(
                "no suitable results present for run_id: {} and testset: {} ".format(
                    self.run_id, testset_name
                ),
                err=True,
            )
            return pandas.DataFrame()

    def get_result_file_path(
        self, testset_name, previous: bool = False
    ) -> Union[str, None]:
        results_dir = os.path.join(self.data_root, "results", self.ftype, testset_name)
        if not os.path.exists(results_dir):
            typer.echo("no results present for {}".format(testset_name), err=True)
            return

        result_files = [
            x
            for x in os.listdir(results_dir)
            if os.path.isfile(os.path.join(results_dir, x))
        ]
        result_files = [
            y for y in result_files if self.run_id in y and y.endswith(self.ftype)
        ]
        result_files.sort(reverse=True)

        idx_f = 1 if previous else 0

        if len(result_files) <= idx_f:
            return
        else:
            return os.path.join(results_dir, result_files[idx_f])

    def load_prev_fullset_report(self):
        fullset_dir = os.path.join(
            self.data_root, "results", "reports", self.fullset_name
        )
        if not os.path.exists(fullset_dir):
            typer.echo("all sets path does not exist", err=True)
            return pandas.DataFrame()

        fullset_files = [
            x
            for x in os.listdir(fullset_dir)
            if x.endswith(self.ftype)
            and "report" in x
            and os.path.isfile(os.path.join(fullset_dir, x))
        ]
        fullset_files = [
            y for y in fullset_files if self.run_id in y and y.endswith(self.ftype)
        ]
        fullset_files.sort()

        if len(fullset_files) < 1:
            typer.echo("there is no previous fullset report")
            return pandas.DataFrame()

        typer.echo(
            "loading previous all sets report: {}".format(
                os.path.join(fullset_dir, fullset_files[-1])
            )
        )
        return pandas.read_csv(
            os.path.join(fullset_dir, fullset_files[-1]), decimal=","
        )

    def load_intent_map(self):
        map_path = os.path.join(self.data_root, "intentmap.csv")
        if os.path.exists(map_path):
            return pandas.read_csv(map_path)
        else:
            typer.echo("could not find intent map", err=True)
            return pandas.DataFrame()

    def get_testset_names(self) -> List[str]:
        testset_names = filter(
            lambda x: x.endswith(self.ftype), os.listdir(self.tset_dir)
        )
        testset_names = [file.replace("." + self.ftype, "") for file in testset_names]

        return testset_names

    def get_resultset_names(self) -> List[str]:
        results_root = os.path.join(self.data_root, "results", self.ftype)
        if not os.path.exists(results_root):
            typer.echo("No results present", err=True)
            return []

        testset_names = [
            x
            for x in os.listdir(results_root)
            if os.path.isdir(os.path.join(results_root, x))
        ]
        testset_names = [
            subdir
            for subdir in testset_names
            if len(os.listdir(os.path.join(results_root, subdir))) > 0
        ]

        return testset_names

    def get_report_names(self) -> List[str]:
        report_root = os.path.join(self.data_root, "results", "reports")
        if not os.path.exists(report_root):
            typer.echo("No reports present", err=True)
            return []

        report_names = [
            x
            for x in os.listdir(report_root)
            if os.path.isdir(os.path.join(report_root, x))
        ]
        report_names = [
            subdir
            for subdir in report_names
            if len(os.listdir(os.path.join(report_root, subdir))) > 0
        ]

        return report_names

    def load_reports(self):
        """
            load latest available reports
            returns tuples of (testset name, DataFrame)
        :return: Iterable[Tuple[str, DataFrame]]
        """
        report_root = os.path.join(self.data_root, "results", "reports")
        if not os.path.exists(report_root):
            typer.echo("No reports present", err=True)
            return []

        report_names = [
            x
            for x in os.listdir(report_root)
            if os.path.isdir(os.path.join(report_root, x))
        ]
        report_names = [
            subdir
            for subdir in report_names
            if len(os.listdir(os.path.join(report_root, subdir))) > 0
        ]
        report_names.sort()

        for report_name in report_names:
            report_files = [
                x
                for x in os.listdir(os.path.join(report_root, report_name))
                if x.endswith(self.ftype) and self.run_id in x and "intent_report" in x
            ]
            if len(report_files) > 0:
                report_files.sort()
                yield report_name, pandas.read_csv(
                    os.path.join(report_root, report_name, report_files[-1]),
                    delimiter=",",
                    decimal=",",
                )
            else:
                typer.echo(
                    "no {} report found for run id {}".format(report_name, self.run_id)
                )

    def load_switch_df(self):
        typer.echo("loading result switch")
        fullset_root = os.path.join(
            self.data_root, "results", "reports", self.fullset_name
        )
        if not os.path.exists(fullset_root):
            typer.echo("No fullset report present", err=True)
            return pandas.DataFrame()

        result_switch_file = next(
            (x for x in os.listdir(fullset_root) if "result_switch" in x), None
        )
        if result_switch_file is None:
            return pandas.DataFrame()

        return pandas.read_csv(os.path.join(fullset_root, result_switch_file))

    def load_conf_matrix_df(self):
        typer.echo("loading confidence matrix")
        fullset_root = os.path.join(
            self.data_root, "results", "reports", self.fullset_name
        )
        if not os.path.exists(fullset_root):
            typer.echo("No fullset report present", err=True)
            return pandas.DataFrame()

        conf_matrix_file = next(
            (x for x in os.listdir(fullset_root) if "conf_matrix.csv" in x), None
        )
        if conf_matrix_file is None:
            return pandas.DataFrame()

        return pandas.read_csv(os.path.join(fullset_root, conf_matrix_file))

    def save_results(self, result: BenchmarkResult):
        typer.echo("saving test results")

        testset_name = result.get_testset_name()
        result_path = os.path.join(
            self.data_root, "results", self.ftype, result.get_testset_name()
        )
        os.makedirs(result_path, exist_ok=True)

        now = datetime.now()
        results_file = os.path.join(
            result_path,
            "{}-{}-{}.{}".format(
                now.strftime("%Y%m%d-%H-%M-%S"),
                result.get_test_name(),
                testset_name,
                self.ftype,
            ),
        )
        typer.echo("to: {}".format(results_file))
        if self.ftype == "csv":
            col_names = [
                "Expected",
                "MATCH",
                "Classified",
                "Confidence",
                "Utterance",
                "Alt1",
                "Alt1_Conf",
                "Alt2",
                "Alt2_Conf",
            ]
            result_rows = []
            for result in result.get_results():
                if len(result.alternatives) >= 3:
                    result_rows.append(
                        [
                            result.intent_label,
                            result.is_match(),
                            result.intent_classification,
                            result.confidence,
                            "{}".format(result.utter),
                            result.alternatives[1]["name"],
                            result.alternatives[1]["confidence"],
                            result.alternatives[2]["name"],
                            result.alternatives[2]["confidence"],
                        ]
                    )
                else:
                    result_rows.append(
                        [
                            result.intent_label,
                            result.is_match(),
                            result.intent_classification,
                            result.confidence,
                            "{}".format(result.utter),
                            "missing",
                            "missing",
                            "missing",
                            "missing",
                        ]
                    )

            dframe = pandas.DataFrame(result_rows, columns=col_names)
            dframe.to_csv(
                results_file, sep=",", float_format="%.3f", decimal=",", index=False
            )

        else:
            typer.echo("Sorry, {} output not implemented yet".format(self.ftype))

        typer.echo("saving done")

    def store_report_zip(self, zip_path: str):
        if zip_path:
            zip_source_path = os.path.join(self.data_root, "results")
            typer.echo("creating zip file")
            t = datetime.now()

            zip_fname = "kw{}_{}_{}".format(
                t.strftime("%U"), self.run_id, t.strftime("%Y%m%d_%H-%M-%S")
            )
            zip_file_path = os.path.join(zip_path, zip_fname)
            shutil.make_archive(
                zip_file_path, "zip", root_dir=zip_source_path
            )  # , base_dir=result_path)
            typer.secho(
                "saved zip to: {}.zip".format(zip_file_path), fg=typer.colors.BLUE
            )

    def results_housekeeping(self):
        """
            clean out old result and report file
        :return:
        """

        typer.echo("cleaning out old reports")
        reports_path = os.path.join(self.data_root, "results", "reports")
        if os.path.exists(reports_path):
            for report_dir in os.listdir(reports_path):
                # keep all_sets reports csv for historical F1
                if report_dir == self.fullset_name:
                    delete_files = [
                        x
                        for x in os.listdir(os.path.join(reports_path, report_dir))
                        if (
                            x.endswith(".png")
                            or "conf_matrix.csv" in x
                            or "result_switch.csv" in x
                        )
                    ]
                    for file in delete_files:
                        print("deleting {}".format(file))
                        os.remove(os.path.join(reports_path, report_dir, file))
                else:
                    shutil.rmtree(os.path.join(reports_path, report_dir))
        else:
            typer.echo("no old reports, nothing to do")
            os.mkdir(reports_path)

        typer.echo("cleaning out old result files")
        results_path = os.path.join(self.data_root, "results", self.ftype)
        if not os.path.exists(results_path) or len(os.listdir(results_path)) == 0:
            typer.echo("there is no results, nothing to do")
            return

        for tset_dir in os.listdir(results_path):
            tset_results_dir = os.path.join(results_path, tset_dir)
            if os.path.isdir(tset_results_dir):
                result_files = [
                    x
                    for x in os.listdir(tset_results_dir)
                    if os.path.isfile(os.path.join(tset_results_dir, x))
                ]
                # only consider result files for given run_id
                result_files = [y for y in result_files if self.run_id in y]
                result_files.sort()
                # keep the last two result files
                to_delete = result_files[:-2]
                for res_file in to_delete:
                    print("deleting {}".format(res_file))
                    os.remove(os.path.join(tset_results_dir, res_file))
            else:
                typer.echo(
                    "ignoring file in top level result folder {}".format(tset_dir)
                )
