import os
import typer

from dotenv import load_dotenv
from google.cloud import dialogflow_v2beta1 as dialogflow

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
SESSION_ID = os.getenv("SESSION_ID")
REGION = os.getenv("REGION", "us")


def load_intents():
    options = {"api_endpoint": REGION + "-dialogflow.googleapis.com"}
    intents_client = dialogflow.IntentsClient(client_options=options)
    parent = (
        dialogflow.AgentsClient.common_location_path(PROJECT_ID, location=REGION)
        + "/agent"
    )
    view = dialogflow.IntentView.INTENT_VIEW_FULL
    intents = intents_client.list_intents(
        request={"parent": parent, "intent_view": view}
    )
    l_intents = []
    for intent in intents:
        l_intents.append(intent)
    return l_intents


def intent_stats():
    intents = load_intents()
    typer.echo(f"Fetching {len(intents)} Intents for Agent {PROJECT_ID} ")
    for intent in intents:
        if len(intent.training_phrases) < 5:
            typer.secho(
                "Intent display_name: {} with {} traingsphrases".format(
                    intent.display_name, len(intent.training_phrases)
                ),
                fg=typer.colors.RED,
            )
        elif len(intent.training_phrases) < 15:
            typer.secho(
                "Intent display_name: {} with {} traingsphrases".format(
                    intent.display_name, len(intent.training_phrases)
                ),
                fg=typer.colors.YELLOW,
            )
        elif len(intent.training_phrases) > 100:
            typer.secho(
                "Intent display_name: {} with {} traingsphrases".format(
                    intent.display_name, len(intent.training_phrases)
                ),
                fg=typer.colors.BRIGHT_YELLOW,
            )
        else:
            typer.secho(
                "Intent display_name: {} with {} traingsphrases".format(
                    intent.display_name, len(intent.training_phrases)
                ),
                fg=typer.colors.GREEN,
            )


def switch_webhook(on: bool):
    typer.echo(f"Setting webhook usage to {on} ")
    from google.protobuf import field_mask_pb2

    parent = (
        dialogflow.AgentsClient.common_location_path(PROJECT_ID, location=REGION)
        + "/agent"
    )
    _intents = load_intents()
    client = dialogflow.IntentsClient(
        client_options={"api_endpoint": REGION + "-dialogflow.googleapis.com"}
    )

    for intent in _intents:
        intent.webhook_state = (
            dialogflow.Intent.WebhookState.WEBHOOK_STATE_ENABLED
            if on
            else dialogflow.Intent.WebhookState.WEBHOOK_STATE_UNSPECIFIED
        )

    update_mask = field_mask_pb2.FieldMask(paths=["webhook_state"])
    batch = dialogflow.IntentBatch()
    batch.intents = _intents
    # @todo this needs review
    response = client.batch_update_intents(
        request={
            "parent": parent,
            "intent_batch_inline": batch,
            "update_mask": update_mask,
        }
    )
    response.add_done_callback(lambda o: print(o.result()))
