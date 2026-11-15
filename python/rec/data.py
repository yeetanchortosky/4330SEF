"""Load the MovieLens subset that the browser trainer uses."""

from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).with_name("movielens.json")


def load_movielens() -> dict:
    with _PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


DATA = load_movielens()
