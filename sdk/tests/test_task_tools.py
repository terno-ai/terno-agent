import json

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools.tasks import (
    InMemoryTaskStore,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
)


@pytest.fixture()
def store():
    return InMemoryTaskStore()


def test_create_and_list(store):
    created = json.loads(TaskCreateTool(store).run(subject="Add feature"))
    assert created["subject"] == "Add feature"
    assert created["status"] == "pending"
    assert created["id"] == "1"

    listed = json.loads(TaskListTool(store).run())
    assert [t["id"] for t in listed] == ["1"]


def test_get_unknown_id(store):
    with pytest.raises(ToolError):
        TaskGetTool(store).run(task_id="99")


def test_update_status_transitions(store):
    TaskCreateTool(store).run(subject="Work")
    updated = json.loads(TaskUpdateTool(store).run(task_id="1", status="in_progress"))
    assert updated["status"] == "in_progress"

    updated = json.loads(TaskUpdateTool(store).run(task_id="1", status="completed"))
    assert updated["status"] == "completed"


def test_update_rejects_invalid_status(store):
    TaskCreateTool(store).run(subject="Work")
    with pytest.raises(ToolError):
        TaskUpdateTool(store).run(task_id="1", status="bogus")


def test_deleted_tasks_hidden_from_list(store):
    TaskCreateTool(store).run(subject="A")
    TaskCreateTool(store).run(subject="B")
    TaskUpdateTool(store).run(task_id="1", status="deleted")
    listed = json.loads(TaskListTool(store).run())
    assert [t["id"] for t in listed] == ["2"]


def test_create_requires_subject(store):
    with pytest.raises(ToolError):
        TaskCreateTool(store).run(subject="   ")


def test_on_change_fires_with_full_list(store):
    seen: list[list[str]] = []
    store.set_on_change(lambda tasks: seen.append([t.status for t in tasks]))

    TaskCreateTool(store).run(subject="A")
    TaskUpdateTool(store).run(task_id="1", status="in_progress")
    TaskUpdateTool(store).run(task_id="1", status="completed")

    assert seen == [["pending"], ["in_progress"], ["completed"]]


def test_on_change_observer_error_does_not_break_mutation(store):
    def boom(_tasks):
        raise RuntimeError("observer blew up")

    store.set_on_change(boom)
    # Mutation must still succeed even though the observer raises.
    created = json.loads(TaskCreateTool(store).run(subject="Resilient"))
    assert created["id"] == "1"
