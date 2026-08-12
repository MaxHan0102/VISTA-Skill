from __future__ import annotations

from types import SimpleNamespace

from vista_skill.integrations.embodiedbench import environment
from vista_skill.integrations.embodiedbench.planner import (
    configure_planner_inference_seed,
)
from vista_skill.models import OpenAICompatibleJsonModel


class RecordingCompletions:
    def __init__(self) -> None:
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content='{"value": true}')
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(prompt_tokens=2, completion_tokens=1)
        return SimpleNamespace(choices=[choice], usage=usage)


def test_planner_proxy_injects_paired_seed_into_executor_request() -> None:
    completions = RecordingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    planner = SimpleNamespace(model=SimpleNamespace(model=client))
    configure_planner_inference_seed(planner, 17)

    planner.model.model.chat.completions.create(model="executor", messages=[])
    assert completions.requests[0]["seed"] == 17


def test_method_model_injects_evolution_seed() -> None:
    completions = RecordingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model = OpenAICompatibleJsonModel("teacher", client=client, seed=23)
    result = model.complete_json(
        system="return JSON",
        content=({"type": "text", "text": "fixture"},),
        schema={
            "type": "object",
            "properties": {"value": {"type": "boolean"}},
            "required": ["value"],
        },
        purpose="seed_test",
    )
    assert result == {"value": True}
    assert completions.requests[0]["seed"] == 23


def test_process_and_habitat_rngs_receive_same_seed(monkeypatch) -> None:
    calls = []

    class FakeNumpyRandom:
        def seed(self, value):
            calls.append(("numpy", value))

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def manual_seed_all(value):
            calls.append(("cuda", value))

    fake_numpy = SimpleNamespace(random=FakeNumpyRandom())
    fake_torch = SimpleNamespace(
        manual_seed=lambda value: calls.append(("torch", value)),
        cuda=FakeCuda(),
    )
    real_import = environment.importlib.import_module

    def fake_import(name):
        if name == "numpy":
            return fake_numpy
        if name == "torch":
            return fake_torch
        return real_import(name)

    monkeypatch.setattr(environment.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        environment.random,
        "seed",
        lambda value: calls.append(("python", value)),
    )
    core = SimpleNamespace(seed=lambda value: calls.append(("habitat", value)))

    environment.seed_process_rngs(31)
    environment.seed_habitat_env(SimpleNamespace(env=core), 31)
    for source in ("python", "numpy", "torch", "cuda", "habitat"):
        assert (source, 31) in calls
