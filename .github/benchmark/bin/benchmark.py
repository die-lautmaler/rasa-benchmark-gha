import concurrent.futures
import os
import pandas
import requests
import typer

from collections import deque
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from google.cloud import dialogflow
from hashlib import md5
from time import time
from typing import List
from uuid import uuid4

RASA_PORT = os.getenv("RASA_PORT", 8005)


class Task:
    """
    A Task represents a Data that is used to conduct a benchmark. It is composed of a utterance
    a expected result, a language.
    Tasks are combined to form a benchmark.
    """

    def __init__(self, utterance: str, intentlabel: str, language: str):
        self.id = md5(utterance.encode())
        self.utterance = utterance
        self.intentlabel = intentlabel
        self.language = language

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False


class TaskResult:
    """
    @Todo: make this an Abstract Class to support different kinds of tasks

    :param utter: utterance under test
    :param confidence: classification confidence
    :param intent_label: expected intent label
    :param intent_classification: classified intent
    :param alternatives: list of alternative classifications
    """

    def __init__(
        self,
        utter: str,
        confidence: float,
        intent_label: str,
        intent_classification: str,
        alternatives: list,
    ):
        self.utter = utter
        self.confidence = confidence
        self.intent_label = intent_label
        self.intent_classification = intent_classification
        self.alternatives = alternatives

    def is_match(self):
        return str.lower(self.intent_label) == str.lower(self.intent_classification)

    def print_colored(self):
        if self.is_match():
            typer.secho(self.__str__(), fg=typer.colors.GREEN)
        else:
            typer.secho(self.__str__(), fg=typer.colors.RED)

    def __str__(self):
        return f"{self.utter} , {self.confidence} , {self.intent_label} ,{self.intent_classification}, {self.is_match()}"


class BenchmarkType(Enum):
    ASR = 1
    NLU_D = 2  # Dialogflow NLU
    NLU_R = 3  # Rasa NLU


class BenchmarkRunState(Enum):
    NEW = 1
    RUNNING = 2
    FAILED = 3
    COMPLETED = 4


class Benchmark:
    """
    A class that takes a list of Tasks and builds a Task Que for a BenchmarkRunner to consume.
    It also stores a reference to a BenchmarkRun so the results of the BenchmarkRun is stored.
    name: dislpay name for the benchamrk
    """

    def __init__(
        self,
        test_name: str,
        testset_name: str,
        benchtype: BenchmarkType,
        tasks: List[Task] = None,
    ):
        self.id = uuid4()
        self.test_name = test_name
        self.testset_name = testset_name
        self.tasks = tasks
        self.benchtype = benchtype
        self.date = datetime.now()
        self._q = deque()

    def task_queue_factory(self):
        """
        for now we only support reading from a csv file in a specific location
        :return: a deque
        """
        for t in self.tasks:
            self._q.append(t)

        return self._q


class BenchmarkRunner:
    """
    BenchmarkRunner loads a collection of Tasks, The Queue will be processed and corresponding results will be stored to disk.
    """

    def __init__(self, benchmark: Benchmark, threads: int = 1):
        self.benchmark = benchmark
        self.threads = threads
        self.result = BenchmarkResult(benchmark.test_name, benchmark.testset_name)
        self.q = benchmark.task_queue_factory()
        self.region = os.getenv("REGION", "us")
        self.client_options = {
            "api_endpoint": self.region + "-dialogflow.googleapis.com"
        }

        if self.benchmark.benchtype == BenchmarkType.ASR:
            raise NotImplementedError("ASR Benchmarks not implemented yet")
        elif self.benchmark.benchtype == BenchmarkType.NLU_D:
            self.worker_function = self.worker_function_dialogflow
        elif self.benchmark.benchtype == BenchmarkType.NLU_R:
            self.worker_function = self.worker_function_rasa
        else:
            raise ValueError("Unknown Benchmark Type")

    def worker_function_rasa(self, t):
        """
            execute classification on rasa server instance
        :param t: classification task
        :return: Benchmark Result
        """
        start_thread_time = time()
        try:
            rasa_response = requests.post(
                f"http://localhost:{RASA_PORT}/model/parse", json={"text": t.utterance}
            )
            if rasa_response.status_code == 200:
                result_json = rasa_response.json()
                res = TaskResult(
                    t.utterance,
                    result_json["intent"]["confidence"],
                    t.intentlabel,
                    result_json["intent"]["name"],
                    result_json["intent_ranking"],
                )
            else:
                print("intent classification request failed")
                print(rasa_response.content)
                res = TaskResult(t.utterance, 0.0, t.intentlabel, "NoResult", [])

        except Exception as e:
            res = TaskResult(t.utterance, 0.0, t.intentlabel, "NoResult", [])
            print("Error when detecting Intent", e)

        thread_time = time() - start_thread_time
        print(f"Duration {thread_time}")
        self.result.add_result(res)
        res.print_colored()
        return res

    def worker_function_dialogflow(self, t):
        start_thread_time = time()
        text_input = dialogflow.TextInput(text=t.utterance, language_code=t.language)

        query_input = dialogflow.QueryInput(text=text_input)

        session_client = dialogflow.SessionsClient(client_options=self.client_options)
        session = (
            "projects/{project}/locations/{region}/agent/sessions/{session}".format(
                project=os.getenv("PROJECT_ID"),
                region=self.region,
                session=self.benchmark.test_name,
            )
        )
        # create result item
        res: TaskResult = None
        try:
            response = session_client.detect_intent(
                request={"session": session, "query_input": query_input}
            )
            res = TaskResult(
                t.utterance,
                response.query_result.intent_detection_confidence,
                t.intentlabel,
                response.query_result.intent_classification.display_name,
                [],
            )
        except Exception as e:
            res = TaskResult(t.utterance, 0.0, t.intentlabel, "NoResult", [])
            print("Error when detecting Intent", e)

        thread_time = time() - start_thread_time
        print(f"Duration {thread_time}")
        self.result.add_result(res)
        res.print_colored()
        return res

    def start(self):
        print("starting benchmark")
        start_time = time()

        # We can use a with statement to ensure threads are cleaned up promptly
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            # Start the load operations and mark each future with its URL
            future_task = {executor.submit(self.worker_function, t): t for t in self.q}
            for future in concurrent.futures.as_completed(future_task):
                res = future.result()

        self.result._state = BenchmarkRunState.COMPLETED
        execution_time = time() - start_time
        self.result.execution_time = execution_time
        print(f"Execution Done in {execution_time}")

    def print_result_summary(self):
        print(self.result)


class BenchmarkResult:
    """
    BenchmarkResult is a data object that stores all results of a given Benchmark.

    """

    def __init__(self, test_name: str, testset_name: str):
        self._test_name = test_name
        self._testset_name = testset_name
        self._state = BenchmarkRunState.NEW
        self._results: List[TaskResult] = []
        self.execution_time = 0

    def is_complete(self):
        if (
            self._state == BenchmarkRunState.FAILED
            or self._state == BenchmarkRunState.COMPLETED
        ):
            return True
        else:
            return False

    def get_test_name(self):
        return self._test_name

    def get_testset_name(self):
        return self._testset_name

    def add_result(self, result: TaskResult):
        self._results.append(result)

    def add_results(self, results: List[TaskResult]):
        self._results.extend(results)

    def get_score(self):
        numnber_of_results = len(self._results)
        confidence = 0
        for i in self._results:
            confidence += i.confidence
        return confidence / numnber_of_results

    def get_results(self):
        return self._results

    def get_number_of_results(self):
        if self._state == BenchmarkRunState.COMPLETED:
            return len(self._results)
        else:
            return -1

    def to_df(self):
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
        for result in self.get_results():
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

        return pandas.DataFrame(result_rows, columns=col_names)

    def get_match_rate(self):
        match = []
        no_match = []
        for i in self._results:
            if i.is_match():
                match.append(i)
            else:
                no_match.append(i)

        return (
            len(self._results),
            len(match),
            len(no_match),
            (len(match) / len(self._results)),
        )

    def __str__(self):
        return f"""Results for Benchmark {self._test_name}: 
State {self._state}
Avg Score  {self.get_score()}
with {self.get_number_of_results()} Samples
"""
