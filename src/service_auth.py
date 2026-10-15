"""Shared authentication for the internal generation and reward services."""
import hmac
import os
import json


def auth_headers():
    token = os.environ.get("SCOPE_SERVICE_TOKEN", "")
    return {"Authorization": "Bearer " + token} if token else {}


def authorized(headers):
    token = os.environ.get("SCOPE_SERVICE_TOKEN", "")
    return bool(token) and hmac.compare_digest(headers.get("Authorization", ""), "Bearer " + token)


def read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if not 0 < length <= 64 * 1024 * 1024:
        raise ValueError("Request body must be between 1 byte and 64 MiB")
    body = json.loads(handler.rfile.read(length))
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body
