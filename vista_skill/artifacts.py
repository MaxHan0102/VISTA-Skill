from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from vista_skill.schemas import dataclass_to_dict


class JsonlArtifactWriter:
    def __init__(self, path: str | os.PathLike[str], *, schema_version: str = "1") -> None:
        self.path = Path(path)
        self.schema_version = schema_version
        self._lock = threading.Lock()

    def append(self, event_type: str, payload: Any) -> None:
        record = {
            "schema_version": self.schema_version,
            "event_type": event_type,
            "payload": dataclass_to_dict(payload),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

