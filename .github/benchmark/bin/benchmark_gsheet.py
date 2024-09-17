# from __future__ import print_function
import datetime
import gspread
import numpy
import os
import pandas as pd
import time

import typer
from googleapiclient.discovery import build
from gspread_dataframe import set_with_dataframe
from gspread_formatting import *
from gspread_formatting.dataframe import *
from oauth2client.service_account import ServiceAccountCredentials
from typing import Iterable, Tuple


class BenchmarkGsheet:
    def __init__(self, folder_id: str, run_id="update"):
        self.run_id = run_id
        self.gclient = self.authenticate()
        self.spreadsheet = self.create_spreadsheet()
        self.quota_timeout = 4
        self.folder_id = folder_id

    def authenticate(self):
        typer.secho("starting google sheets client", fg=typer.colors.BLUE)
        gclient = gspread.service_account(os.environ.get("GOOGLE_SHEETS_CREDENTIALS"))
        return gclient

    def create_spreadsheet(self):
        now = datetime.datetime.now()
        spreadsheet = self.gclient.create(
            "kw{:02d}_{}_report".format(now.isocalendar()[1], self.run_id)
        )

        return spreadsheet

    def upload_worksheet(self, worksheet_name: str, df: pd.DataFrame):
        worksheet = self.spreadsheet.add_worksheet(worksheet_name, 0, 0)
        set_with_dataframe(worksheet, df)
        return worksheet

    def move_spreadsheet(self):
        typer.secho("moving spreadsheet", fg=typer.colors.GREEN)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            os.environ.get("GOOGLE_SHEETS_CREDENTIALS"), scopes
        )
        service_drive = build("drive", "v3", credentials=creds)
        file_id = self.spreadsheet.id
        file = service_drive.files().get(fileId=file_id, fields="parents").execute()
        previous_parents = ",".join(file.get("parents"))
        service_drive.files().update(
            fileId=file_id,
            addParents=self.folder_id,
            removeParents=previous_parents,
            fields="id, parents",
        ).execute()

    def nlu_update_report(
        self,
        df_results: pd.DataFrame,
        reports_generator: Iterable[Tuple[str, pd.DataFrame]],
        df_result_switch: pd.DataFrame,
        conf_matrix_df: pd.DataFrame,
    ):
        self.create_fullset_results_worksheet(df_results)
        report_summaries = self.create_report_worksheets(reports_generator)
        self.create_switch_worksheet(df_result_switch)
        self.create_overview_worksheet(report_summaries)
        self.create_conf_matrix_worksheet(conf_matrix_df)

    def create_fullset_results_worksheet(self, df_results: pd.DataFrame):
        """
        # upload fullset individual results worksheet
        """
        if df_results.empty:
            typer.echo("not adding individual results", err=True)
        else:
            df_results.sort_values(
                by=["MATCH", "Expected", "Classified", "Confidence"], inplace=True
            )

            typer.secho("uploading individual results", fg=typer.colors.BLUE)
            self.upload_worksheet("all_sets_results", df_results)

            typer.secho(
                "\u001b[38;5;245mwaiting {} seconds to avoid api quota timeout".format(
                    self.quota_timeout
                )
            )
            time.sleep(self.quota_timeout)

    def create_report_worksheets(
        self, reports_generator: Iterable[Tuple[str, pd.DataFrame]]
    ):
        """
        # create report worksheets
        """
        report_summaries = []
        for testset_name, df_report in reports_generator:
            # collect values for overview sheet
            all_values = df_report.loc[df_report["Intent"] == "ALL"].copy()
            all_values["Intent"] = testset_name
            makro_values = df_report.loc[df_report["Intent"] == "ALL_MAKRO"].copy()
            makro_values["Intent"] = "{}_MAKRO".format(testset_name)
            report_summaries.extend([all_values, makro_values])

            df_report.sort_values(
                ["Samples", "Fallbacks", "F1 Score"], ascending=True, inplace=True
            )
            typer.secho(
                "uploading {} report".format(testset_name), fg=typer.colors.BLUE
            )
            ws_report = self.upload_worksheet(
                testset_name + "_intent_report", df_report
            )

            if testset_name == "all_sets":
                reqs = {
                    "requests": [
                        {
                            "updateSheetProperties": {
                                "properties": {"index": 1, "sheetId": ws_report.id},
                                "fields": "index",
                            }
                        }
                    ]
                }
                self.spreadsheet.batch_update(body=reqs)

            typer.echo("format report")
            formatter = BasicFormatter(decimal_format="0.000", freeze_headers=True)
            format_with_dataframe(
                ws_report,
                df_report,
                formatter,
                include_index=False,
                include_column_header=True,
            )
            ws_report.hide_columns(9, 10)
            self.reports_overview_conditional_rules(ws_report)

            typer.secho(
                "\u001b[38;5;245mwaiting {} seconds to avoid api quota timeout".format(
                    self.quota_timeout
                )
            )
            time.sleep(self.quota_timeout)

        # delete initial default worksheet
        self.spreadsheet.del_worksheet(self.spreadsheet.worksheets()[0])

        return report_summaries

    def create_switch_worksheet(self, df_result_switch: pd.DataFrame):
        """
        # upload the classification switch worksheet
        """
        if df_result_switch.empty:
            typer.secho("No classification switches present", fg=typer.colors.YELLOW)
        else:
            typer.secho("uploading classification switches", fg=typer.colors.BLUE)
            ws_switch = self.upload_worksheet(
                "all_sets_classification_switch", df_result_switch
            )

            typer.echo("format switches")
            formatter = BasicFormatter(
                freeze_headers=True,
            )
            format_with_dataframe(
                ws_switch,
                df_result_switch,
                formatter,
                include_index=False,
                include_column_header=True,
            )
            set_column_width(ws_switch, "A:C", 260)
            self.switch_conditional_rules(ws_switch)

        typer.secho(
            "\u001b[38;5;245mwaiting {} seconds to avoid api quota timeout".format(
                self.quota_timeout
            )
        )
        time.sleep(self.quota_timeout)

    def create_overview_worksheet(self, report_summaries):
        """
        # create overview worksheet
        """
        try:
            df_summary = pd.concat(report_summaries)
            typer.secho("uploading reports overview", fg=typer.colors.BLUE)
            ws_overview = self.upload_worksheet("testsets-overview", df_summary)

            typer.echo("format reports overview ")
            # not really adapted for overview sheet but still good:
            self.reports_overview_conditional_rules(ws_overview)
        except gspread.exceptions.APIError as error:
            typer.echo(
                "Could NOT create overview sheet: {}".format(error.response.text),
                err=True,
            )

    def create_conf_matrix_worksheet(self, conf_matrix_df: pd.DataFrame):
        """
        # create confusion matrix worksheet
        """
        if conf_matrix_df.empty:
            typer.echo("not adding confusion matrix", err=True)
        else:
            conf_matrix_nan = conf_matrix_df.replace(0, numpy.nan)
            typer.secho("uploading confusion matrix", fg=typer.colors.BLUE)
            ws_matrix = self.upload_worksheet(
                "all_sets_confusion_matrix", conf_matrix_nan
            )

            typer.echo("format confusion matrix")
            self.conf_matrix_conditional_rules(ws_matrix)

            typer.secho(
                "\u001b[38;5;245mwaiting {} seconds to avoid api quota timeout".format(
                    self.quota_timeout
                )
            )
            time.sleep(self.quota_timeout)

    def reports_overview_conditional_rules(self, worksheet):
        rules = get_conditional_format_rules(worksheet)
        row_count = str(len(worksheet.get_all_values()))

        rule_samples = ConditionalFormatRule(
            ranges=[GridRange.from_a1_range("B1:B" + row_count, worksheet)],
            gradientRule=GradientRule(
                minpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(230),
                        self.rgb_to_srgb(124),
                        self.rgb_to_srgb(115),
                    ),
                    type="MIN",
                ),
                midpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(251),
                        self.rgb_to_srgb(188),
                        self.rgb_to_srgb(4),
                    ),
                    type="NUMBER",
                    value="15",
                ),
                maxpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(87),
                        self.rgb_to_srgb(187),
                        self.rgb_to_srgb(138),
                    ),
                    type="NUMBER",
                    value="40",
                ),
            ),
        )
        rules.append(rule_samples)

        rule_fallbacks = ConditionalFormatRule(
            ranges=[GridRange.from_a1_range("E1:E" + row_count, worksheet)],
            gradientRule=GradientRule(
                minpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(255),
                        self.rgb_to_srgb(255),
                        self.rgb_to_srgb(255),
                    ),
                    type="NUMBER",
                    value="0.1",
                ),
                maxpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(230),
                        self.rgb_to_srgb(124),
                        self.rgb_to_srgb(115),
                    ),
                    type="NUMBER",
                    value="0.2",
                ),
            ),
        )
        rules.append(rule_fallbacks)

        rule_f1 = ConditionalFormatRule(
            ranges=[GridRange.from_a1_range("H1:H" + row_count, worksheet)],
            gradientRule=GradientRule(
                minpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(230),
                        self.rgb_to_srgb(124),
                        self.rgb_to_srgb(115),
                    ),
                    type="NUMBER",
                    value="0.1",
                ),
                midpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(251),
                        self.rgb_to_srgb(188),
                        self.rgb_to_srgb(4),
                    ),
                    type="NUMBER",
                    value="0.65",
                ),
                maxpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(87),
                        self.rgb_to_srgb(187),
                        self.rgb_to_srgb(138),
                    ),
                    type="NUMBER",
                    value="0.95",
                ),
            ),
        )
        rules.append(rule_f1)

        rule_all_sets_diff_neg = ConditionalFormatRule(
            ranges=[GridRange.from_a1_range("I1:I" + row_count, worksheet)],
            booleanRule=BooleanRule(
                condition=BooleanCondition("NUMBER_LESS", ["0"]),
                format=CellFormat(
                    backgroundColor=Color(
                        self.rgb_to_srgb(230),
                        self.rgb_to_srgb(124),
                        self.rgb_to_srgb(115),
                    )
                ),
            ),
        )
        rules.append(rule_all_sets_diff_neg)

        rule_all_sets_diff_pos = ConditionalFormatRule(
            ranges=[GridRange.from_a1_range("I1:I" + row_count, worksheet)],
            booleanRule=BooleanRule(
                condition=BooleanCondition("NUMBER_GREATER", ["0"]),
                format=CellFormat(
                    backgroundColor=Color(
                        self.rgb_to_srgb(87),
                        self.rgb_to_srgb(187),
                        self.rgb_to_srgb(138),
                    )
                ),
            ),
        )
        rules.append(rule_all_sets_diff_pos)

        rules.save()

    def conf_matrix_conditional_rules(self, worksheet):
        rules = get_conditional_format_rules(worksheet)
        row_count = str(len(worksheet.get_all_values()))
        rule = ConditionalFormatRule(
            ranges=[GridRange.from_a1_range("1:" + row_count, worksheet)],
            gradientRule=GradientRule(
                minpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(185),
                        self.rgb_to_srgb(240),
                        self.rgb_to_srgb(133),
                    ),
                    type="MIN",
                ),
                midpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(17), self.rgb_to_srgb(183), self.rgb_to_srgb(0)
                    ),
                    type="PERCENTILE",
                    value="50",
                ),
                maxpoint=InterpolationPoint(
                    color=Color(
                        self.rgb_to_srgb(17), self.rgb_to_srgb(134), self.rgb_to_srgb(0)
                    ),
                    type="MAX",
                ),
            ),
        )
        rules.append(rule)
        rules.save()

    def switch_conditional_rules(self, worksheet):
        rules = get_conditional_format_rules(worksheet)
        row_count = str(len(worksheet.get_all_values()))
        rule_true_false = ConditionalFormatRule(
            ranges=[GridRange.from_a1_range("1:" + row_count, worksheet)],
            booleanRule=BooleanRule(
                condition=BooleanCondition("TEXT_EQ", ["FALSCH-WAHR"]),
                format=CellFormat(
                    backgroundColor=Color(
                        self.rgb_to_srgb(161),
                        self.rgb_to_srgb(243),
                        self.rgb_to_srgb(177),
                    )
                ),
            ),
        )
        rules.append(rule_true_false)

        rule_false_true = ConditionalFormatRule(
            ranges=[GridRange.from_a1_range("1:" + row_count, worksheet)],
            booleanRule=BooleanRule(
                condition=BooleanCondition("TEXT_EQ", ["WAHR-FALSCH"]),
                format=CellFormat(
                    backgroundColor=Color(
                        self.rgb_to_srgb(239),
                        self.rgb_to_srgb(161),
                        self.rgb_to_srgb(161),
                    )
                ),
            ),
        )
        rules.append(rule_false_true)
        rules.save()

    def rgb_to_srgb(self, x: float) -> float:
        # first, convert rgb to linear
        x = pow(x / 255, 2.2)
        # then, convert to srgb
        if x <= 0.0:
            return 0.0
        elif x >= 1:
            return 1.0
        elif x < 0.0031308:
            return x * 12.92
        else:
            return x ** (1 / 2.4) * 1.055 - 0.055

    if __name__ == "__main__":
        print("please use the benchmark gsheet command")
