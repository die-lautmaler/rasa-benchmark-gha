import os
from datetime import datetime
from typing import List

import numpy
import numpy as np
import pandas as pd
import seaborn
import typer
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class BenchmarkReport:
    """
    A class that accumulates some summarizing numbers from individual test results
    provides confusion matrix computation and plotting
    per intent and overall F1 scores, list of errors
    """

    def __init__(self, benchmark_result: pd.DataFrame, testset_name: str, run_id: str):
        """
        :param benchmark_result: a pd dataframe with the results
        :param testset_name: the name of the testset, e.g. "account", "support", ...
        :param run_id: the run_id, default "benchmark"
        """
        self.benchmark_result = benchmark_result
        self.testset_name = testset_name
        self.run_id = run_id
        self.conf_matrix = pd.DataFrame()

    def create_report(self, out_path):
        os.makedirs(out_path, exist_ok=True)
        tstamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")

        print("creating confusion matrix")
        self.compute_confusion_matrix(self.benchmark_result)
        plot = self.plot_confusion_matrix()
        plot.savefig(
            os.path.join(out_path, tstamp + "-" + self.run_id + "-conf_matrix.png"),
            bbox_inches="tight",
            dpi=300,
        )

        print("creating intent report")
        intent_report = self.intent_report()

        intent_report.to_csv(
            os.path.join(out_path, tstamp + "-" + self.run_id + "-intent_report.csv"),
            sep=",",
            float_format="%.3f",
            decimal=",",
            index=False,
        )

        print("creating confidence histogram")
        plot = self.plot_confidence_hist()
        fig = plot.gcf()
        fig.set_size_inches(16, 8)
        fig.savefig(
            os.path.join(out_path, tstamp + "-" + self.run_id + "-conf_hist.png"),
            bbox_inches="tight",
            dpi=300,
        )

    def create_fullset_report(
        self,
        out_path: str,
        df_prev_report: pd.DataFrame,
        df_prev_results: pd.DataFrame,
        df_intent_map: pd.DataFrame,
    ):
        os.makedirs(out_path, exist_ok=True)
        tstamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")

        typer.echo("creating confusion matrix")
        self.compute_confusion_matrix(self.benchmark_result)
        fpath_confmatrix_csv = os.path.join(
            out_path, "{}-{}-conf_matrix.csv".format(tstamp, self.run_id)
        )
        self.conf_matrix.to_csv(fpath_confmatrix_csv)

        typer.echo("plotting confusion matrix")
        plot = self.plot_confusion_matrix()
        fpath_confmatrix_png = os.path.join(
            out_path, "{}-{}-conf_matrix.png".format(tstamp, self.run_id)
        )
        plot.savefig(fpath_confmatrix_png, bbox_inches="tight", dpi=300)

        result_switch = pd.DataFrame()
        if df_prev_results.empty:
            typer.echo("skipping result switch computation, no previous results")
        else:
            typer.echo("creating result switch")
            result_switch = self.compute_switch_results(df_prev_results)

            fpath_result_switch = os.path.join(
                out_path, "{}-{}-result_switch.csv".format(tstamp, self.run_id)
            )
            result_switch.to_csv(
                fpath_result_switch,
                sep=",",
                float_format="%.3f",
                decimal=",",
                index=False,
            )

        typer.echo("creating intent report")
        df_report = self.intent_report(df_intent_map)

        if df_prev_report.empty:
            typer.echo("Skipping F1 diff to previous run")
        else:
            typer.echo("computing F1 diff to previous run")
            merged_f1 = pd.merge(
                df_report[["F1 Score", "Intent ID", "Diff"]],
                df_prev_report[["F1 Score", "Intent ID"]],
                on="Intent ID",
                how="left",
            )

            # add 0. to avoid -0.000 after rounding
            merged_f1["Diff"] = (
                numpy.around(merged_f1["F1 Score_x"] - merged_f1["F1 Score_y"], 3) + 0.0
            )
            df_report["Diff"] = merged_f1["Diff"]

        fpath_intent_report = os.path.join(
            out_path, "{}-{}-intent_report.csv".format(tstamp, self.run_id)
        )
        df_report.to_csv(
            fpath_intent_report, sep=",", float_format="%.3f", decimal=",", index=False
        )

        typer.echo("creating confidence histogram")
        fpath_hist = os.path.join(
            out_path, "{}-{}-conf_hist.png".format(tstamp, self.run_id)
        )
        plot = self.plot_confidence_hist()
        fig = plot.gcf()
        fig.set_size_inches(16, 8)
        fig.savefig(fpath_hist, bbox_inches="tight", dpi=300)

    def compute_confusion_matrix(self, df_results: pd.DataFrame):
        """
            compute the confusion matrix from a results dataframe
        :param df_results:
        :return:
        """
        intents_expected = df_results["Expected"].unique()
        # get intents that are not expected at all
        extra_intents_classified = df_results.loc[
            ~df_results["Classified"].isin(intents_expected), "Classified"
        ].unique()

        # make the matrix symmectrical for expected intents
        intents_expected.sort()
        extra_intents_classified.sort()
        intents_classified = np.concatenate(
            [intents_expected, extra_intents_classified]
        )

        conf_matrix = pd.crosstab(df_results["Expected"], df_results["Classified"])
        conf_matrix = conf_matrix.reindex(
            index=intents_expected, columns=intents_classified, fill_value=0
        )

        self.conf_matrix = conf_matrix

        return conf_matrix

    def plot_confusion_matrix(self, interactive=False):
        if self.conf_matrix.size < 1:
            self.compute_confusion_matrix(self.benchmark_result)

        matrix_with_nan = self.conf_matrix.replace(0, numpy.nan)

        seaborn.set(color_codes=True)
        plt.figure(
            1,
            figsize=(
                5 + (0.35 * len(matrix_with_nan.columns.values)),
                3 + (0.35 * len(matrix_with_nan.index.values)),
            ),
        )

        if len(self.run_id) > 0:
            plt.title("Confusion Matrix - " + self.run_id + "-" + self.testset_name)
        else:
            plt.title("Confusion Matrix - " + self.testset_name)

        seaborn.set(font_scale=1.3)
        ax = seaborn.heatmap(
            matrix_with_nan,
            annot=True,
            linewidths=0.2,
            linecolor="black",
            cbar=False,
            cmap="YlGnBu",
            xticklabels=True,
            yticklabels=True,
        )  # cbar_kws={'label': 'Scale'},

        ax.set_xticklabels(matrix_with_nan.columns.values, rotation=45, ha="right")
        ax.set_yticklabels(matrix_with_nan.index.values)

        ax.set(ylabel="Expected Label", xlabel="Classified Label")

        if interactive:
            plt.show()

        return plt

    def compute_switch_results(self, df_results_previous: pd.DataFrame):
        df_results_current = self.benchmark_result
        switch_data = []
        for _, result_current in df_results_current.iterrows():
            utterance = result_current["Utterance"]
            df_prev_row = df_results_previous.loc[
                df_results_previous["Utterance"] == utterance
            ]
            if df_prev_row.empty:
                # print(f"This utterance was not in the previous test set: {utterance}")
                continue
            else:
                series_prev = df_prev_row.squeeze()
                match_current = result_current["MATCH"]
                match_prev = series_prev["MATCH"]
                classified_current = result_current["Classified"]
                classified_prev = series_prev["Classified"]

                try:
                    if match_current and match_prev:
                        # True this and previous week
                        continue
                    else:
                        if match_current:
                            switch = "FALSCH-WAHR"
                        elif match_prev:
                            switch = "WAHR-FALSCH"
                        else:
                            switch = "FALSCH-FALSCH"
                            if classified_current == classified_prev:
                                continue

                        switch_data.append(
                            [
                                result_current["Expected"],
                                classified_prev,
                                classified_current,
                                switch,
                                utterance,
                            ]
                        )
                except ValueError as e:
                    print(e)
                    continue

        df_switch = pd.DataFrame(
            data=switch_data,
            columns=[
                "Expected",
                "Classified last week",
                "Classified this week",
                "Switch",
                "Utterance",
            ],
        )
        df_switch.sort_values(
            by=["Switch", "Expected"], ascending=[False, True], inplace=True
        )

        return df_switch

    """
    # this method expects dataframe with results from a nlu model without a fallback classifier
    # it computes an what if analysis on the impact of the fallback threshold value 
    # it starts with a threshold of 1.0 and decreases it by step_size in every iteration until 0
    # for every step it computes the weighted ratio between correct / (false * 2.5) + fallback classifications
    # as can be seen a misclassification is weighted 2.5 times more than an nlu fallback
    # the measure is user centric in so far, that an nlu fallback can be handled gracefully, whereas a
    # misclassification stays unidentified and is much more frustrating for the user
    # it returns a dataframe that contains the main parameters of each steps computations in its rows  
    """

    @staticmethod
    def compute_optimal_fb_thresh(
        df_results: pd.DataFrame, step_size: float
    ) -> pd.DataFrame:
        step_results = []
        sample_size = df_results.shape[0]
        current_th = 1.0
        # number of correct/incorrect classifications above threshold
        correct = 0
        incorrect = 0

        while current_th >= 0.0:
            step_mask = (df_results["Confidence"] >= current_th) & (
                df_results["Confidence"] < (current_th + step_size)
            )

            df_interval = df_results.loc[step_mask]
            # print(
            #     "{:.2f} - {:.2f} count".format(current_th, current_th + step_size),
            #     df_interval.shape[0],
            # )
            correct += df_interval[df_interval.MATCH == True].shape[0]
            incorrect += df_interval[df_interval.MATCH == False].shape[0]

            # print("True",  df_interval[df_interval.MATCH == True].shape[0])
            # print("False", df_interval[df_interval.MATCH == False].shape[0])

            fb_count = sample_size - (correct + incorrect)
            interval_ratio_weighted = correct / ((incorrect * 2.5) + fb_count)
            interval_ratio = correct / incorrect
            classified_proportion = (correct + incorrect) / sample_size
            # print("ratio: {:.3f}".format(interval_ratio))
            step_results.append(
                [
                    current_th,
                    round(interval_ratio, 3),
                    round(interval_ratio_weighted, 3),
                    round(classified_proportion, 3),
                ]
            )
            current_th -= step_size
            current_th = round(current_th, 4)

        # step_results.sort()
        # print(step_results)

        return pd.DataFrame(
            data=step_results,
            columns=[
                "Fallback Threshold",
                "Ratio Correct/False Pure",
                "Ratio Correct/False Fallback weighted",
                "Classified Samples Proportion",
            ],
        )

    def _compute_confidences(self):
        """
            computes a list of two lists containing the confidence values for correct / false classifications of a benchmark
        :return: [[confs_correct], [confs_false]]
        """
        confs_correct: List[float] = []
        confs_false: List[float] = []

        confs_correct = self.benchmark_result.loc[
            self.benchmark_result["Expected"] == self.benchmark_result["Classified"],
            "Confidence",
        ]
        confs_false = self.benchmark_result.loc[
            self.benchmark_result["Expected"] != self.benchmark_result["Classified"],
            "Confidence",
        ]

        return [confs_correct.tolist(), confs_false.tolist()]

    def plot_confidence_hist(self, interactive=False):
        hist_data = self._compute_confidences()
        plt.gcf().clear()

        # Wine-ish colour for the confidences of hits.
        # Blue-ish colour for the confidences of misses.
        colors = ["#009292", "#920000"]

        # total range for the confidence intervals tobe displayed
        max_value = 1.00
        min_value = 0.55

        # bin_width = (max_value - min_value) / n_bins
        # n_bins = int((max_value - min_value) * 10 * 2)
        n_bins = 21
        # print("max", max_value, "min", min_value, "bins", n_bins)
        bin_width = 0.02
        bins = [round(min_value + (i * bin_width), 2) for i in range(1, n_bins + 1)]
        # add true/false confidences to the histogram
        binned_data_sets = [numpy.histogram(d, bins=bins)[0] for d in hist_data]

        max_xlims = [max(binned_data_set) for binned_data_set in binned_data_sets]
        max_xlims = [xlim + numpy.ceil(0.25 * xlim) for xlim in max_xlims]  # padding

        min_ylim = (
            bins[
                min(
                    [
                        (binned_data_set != 0).argmax(axis=0)
                        for binned_data_set in binned_data_sets
                    ]
                )
            ]
            - bin_width
        )

        max_ylim = max(bins) + bin_width

        yticks = [float("{:.2f}".format(x)) for x in bins]
        centers = 0.5 * (0.04 + (bins + numpy.roll(bins, 0))[:-1])
        heights = 0.85 * numpy.diff(bins)

        fig, axes = plt.subplots(ncols=2, sharey="all")
        axes[0].barh(
            centers,
            binned_data_sets[0],
            height=heights,
            align="center",
            color=colors[0],
            label="hits",
        )
        axes[0].set(title="Correct")
        axes[0].grid(True, which="both")
        # axes[0].tick_params(which='major', length=4, color='r')

        axes[1].barh(
            centers,
            binned_data_sets[1],
            height=heights,
            align="center",
            color=colors[1],
            label="misses",
        )
        axes[1].set(title="Wrong")
        axes[1].xaxis.grid(True, which="both")

        axes[0].set(yticks=yticks, xlim=(0, max_xlims[0]), ylim=(min_ylim, max_ylim))
        axes[1].set(yticks=yticks, xlim=(0, max_xlims[1]), ylim=(min_ylim, max_ylim))

        axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        axes[0].invert_xaxis()
        axes[0].yaxis.tick_right()

        fig.subplots_adjust(
            wspace=0.1
        )  # get the graphs exactly far enough apart for yaxis labels
        title = "Confidence Histogram - " + self.run_id
        fig.suptitle(title, fontsize="large", fontweight="bold")

        # Add hidden plot to correctly add x and y labels
        fig.add_subplot(111, frameon=False)
        # Hide tick and tick label of the big axis
        plt.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
        plt.grid(False, which="both")
        plt.ylabel("Confidence")
        plt.xlabel("Number of Samples")

        if interactive:
            plt.show()

        return plt

    def intent_report(self, df_intent_map: pd.DataFrame = pd.DataFrame()):
        """
            computes a table for per intent samples count/count correct/count false/precision/recall/F1 Score
            one row for summarized values of the whole testset
        :return: a pd dataframe with the intent report
        """
        col_labels = [
            "Intent",
            "Samples",
            "Correct",
            "False",
            "Fallbacks",
            "Precision",
            "Recall",
            "F1 Score",
            "Diff",
            "Intent ID",
        ]
        # index_labels = ["ALL"] + self.conf_matrix.index.values
        report_data = []

        sample_sizes = self.conf_matrix.sum(axis=1)
        samples_size_total = sample_sizes.sum()
        hit_counts = self.conf_matrix.sum(axis=0)
        true_pos_sum = 0
        false_pos_sum = 0
        false_neg_sum = 0

        prec_scores = []
        rec_scores = []

        for intent in self.conf_matrix.columns:
            fallback_rate = 0.0
            try:
                fallback_rate = (
                    float(self.conf_matrix.loc[intent, "nlu_fallback"])
                    / sample_sizes[intent]
                )
            except KeyError:
                pass
            except IndexError:
                pass

            try:
                true_pos = self.conf_matrix.loc[intent, intent]

                false_pos = hit_counts[intent] - true_pos
                false_neg = sample_sizes[intent] - true_pos

                true_pos_sum += true_pos
                false_pos_sum += false_pos
                false_neg_sum += false_neg

                prec = 0.0
                reca = 0.0
                f_one = 0.0
                if true_pos > 0:
                    prec = true_pos / (true_pos + false_pos)
                    reca = true_pos / (true_pos + false_neg)
                    f_one = true_pos / (true_pos + (0.5 * (false_pos + false_neg)))

                prec_scores.append(prec)
                rec_scores.append(reca)

                intent_report = [
                    intent,
                    int(sample_sizes[intent]),
                    int(true_pos),
                    int(false_neg),
                    fallback_rate,
                    prec,
                    reca,
                    f_one,
                    0.0,
                    "",
                ]
                report_data.append(intent_report)
            except KeyError:
                # intent is not under the expected ones, so all classifications are false positives
                # print("unexpected intent")
                false_pos_sum += hit_counts[intent]
                prec_scores.append(0.0)
                rec_scores.append(0.0)

        fallback_rate_sum = 0.0
        try:
            fallback_rate_sum = hit_counts["nlu_fallback"] / sample_sizes.sum()
        except KeyError:
            pass
        except IndexError:
            pass

        prec_all = true_pos_sum / (true_pos_sum + false_pos_sum)
        reca_all = true_pos_sum / (true_pos_sum + false_neg_sum)
        f_one_all = true_pos_sum / (
            true_pos_sum + (0.5 * (false_pos_sum + false_neg_sum))
        )

        report_data.append(
            [
                "ALL",
                int(sample_sizes.sum()),
                int(true_pos_sum),
                int(false_neg_sum),
                fallback_rate_sum,
                prec_all,
                reca_all,
                f_one_all,
                0.0,
                "ALL",
            ]
        )

        # mean over all individual scores
        prec_all_macro = numpy.mean(prec_scores)
        reca_all_macro = numpy.mean(rec_scores)
        f_one_all_macro = (
            2 * (prec_all_macro * reca_all_macro) / (prec_all_macro + reca_all_macro)
        )

        report_data.append(
            [
                "ALL_MAKRO",
                int(sample_sizes.sum()),
                int(true_pos_sum),
                int(false_neg_sum),
                fallback_rate_sum,
                prec_all_macro,
                reca_all_macro,
                f_one_all_macro,
                0.0,
                "ALL_MAKRO",
            ]
        )

        # if expected intent is not classified at all, NaN values occur
        # so they get replaced with zeros
        # return pd.DataFrame(report_data, columns=col_labels).replace(numpy.nan, 0)
        report_df = pd.DataFrame(report_data, columns=col_labels)
        # add Intent IDs
        if not df_intent_map.empty:
            merged = pd.merge(
                report_df["Intent"],
                df_intent_map,
                left_on="Intent",
                right_on="displayName",
                how="left",
            )
            report_df["Intent ID"] = merged["name"]

        return report_df
