#!/usr/bin/env python3

import typer
import csv
import os
import yaml

from typing import Optional, List

# from dotenv import load_dotenv

app = typer.Typer()
data = {}
data_dir = "data"


def convert_test_set(ts_file, ftype: str = "csv"):
    test_set_csv = os.path.join("..", data_dir, ftype, ts_file + "." + ftype)

    if not os.path.isfile(test_set_csv):
        typer.secho(
            "Please provide a a valid testset file", err=True, fg=typer.colors.RED
        )
        raise ValueError("no valid testfile found")
    else:
        with open(test_set_csv, "r", encoding="utf8") as ts_file:
            for line in csv.reader(ts_file):
                if line[1]:
                    print(line)
                    add_data(line[1], line[0])
                else:
                    print("unlabeled line")


def add_data(intent: str, utterance: str):
    if intent in data:
        # add to list
        data[intent].append(utterance)
    else:
        # create a new list
        data[intent] = [utterance]


def serialize_rasa_testset(testset):
    testset_path = os.path.join("..", "data", "rasa")
    if not (os.path.exists(testset_path)):
        os.makedirs(testset_path)

    testset_file = os.path.join(testset_path, testset + ".yml")
    print("writing testset {} yaml to {}".format(testset, testset_file))
    with open(testset_file, "w", encoding="utf8") as yaml_file:
        yaml_file.write('version: "2.0"\nnlu:\n')
        for dict_key in data:
            yaml_file.write(f"- intent: {dict_key}\n")
            yaml_file.write("  examples: |\n")
            for utterance in data[dict_key]:
                utterance = utterance.replace("\n", " ")
                yaml_file.write(f"    - {str.lower(str.strip(utterance))}\n")


@app.command()
def create_test_set(
    testsets: List[str],
    out_testset: str = typer.Option(None, help="set the name of the resulting testset"),
    ftype: Optional[str] = "csv",
):
    typer.echo("implement me")
    typer.echo(f"Converting and aggregating {len(testsets)} data sets. ")
    for testset in testsets:
        convert_test_set(testset, ftype)
    typer.echo("Need to save this")
    typer.echo(yaml.dump(data))
    if not out_testset:
        out_testset = "_".join(testsets)

    serialize_rasa_testset(out_testset)
    typer.echo("Rasa Test Set created")


if __name__ == "__main__":
    app()
