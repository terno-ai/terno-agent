import json

import pytest

from terno_agent.core.exceptions import ToolError
from terno_agent.tools.tasks import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskStore,
    TaskUpdateTool,
)


@pytest.fixture()
def store():
    return TaskStore()


def test_create_and_list(store):
    created = json.loads(TaskCreateTool(store).run(subject="Add feature"))
    assert created["subject"] == "Add feature"
    assert created["status"] == "pending"
    assert created["id"] == "1"

    listed = json.loads(TaskListTool(store).run())
    assert [t["id"] for t in listed] == ["1"]


def test_get_unknown_id(store):
    with pytest.raises(ToolError):
        TaskGetTool(store).run(taskId="99")


def test_update_status_transitions(store):
    TaskCreateTool(store).run(subject="Work")
    updated = json.loads(TaskUpdateTool(store).run(taskId="1", status="in_progress"))
    assert updated["status"] == "in_progress"

    updated = json.loads(TaskUpdateTool(store).run(taskId="1", status="completed"))
    assert updated["status"] == "completed"


def test_update_rejects_invalid_status(store):
    TaskCreateTool(store).run(subject="Work")
    with pytest.raises(ToolError):
        TaskUpdateTool(store).run(taskId="1", status="bogus")


def test_deleted_tasks_hidden_from_list(store):
    TaskCreateTool(store).run(subject="A")
    TaskCreateTool(store).run(subject="B")
    TaskUpdateTool(store).run(taskId="1", status="deleted")
    listed = json.loads(TaskListTool(store).run())
    assert [t["id"] for t in listed] == ["2"]


def test_create_requires_subject(store):
    with pytest.raises(ToolError):
        TaskCreateTool(store).run(subject="   ")


def test_ported_task_tool_names_and_params():
    # Names and parameter spellings come from the captured schemas.
    store = TaskStore()
    assert TaskCreateTool(store).schema.name == "TaskCreate"
    assert TaskListTool(store).schema.name == "TaskList"
    assert TaskGetTool(store).schema.name == "TaskGet"
    assert TaskUpdateTool(store).schema.name == "TaskUpdate"

    # camelCase, matching the capture — not Terno's old snake_case.
    assert "activeForm" in TaskCreateTool(store).schema.parameters["properties"]
    for tool in (TaskGetTool(store), TaskUpdateTool(store)):
        assert tool.schema.parameters["required"] == ["taskId"]
