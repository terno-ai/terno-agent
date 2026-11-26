import json

import pytest

from terno_agent.agents.terno import TernoAgent
from terno_agent.core.exceptions import ToolError
from terno_agent.tools.schedules import (
    InMemoryScheduleStore,
    ScheduleCreateTool,
    ScheduleDeleteTool,
    ScheduleGetTool,
    ScheduleListTool,
    ScheduleUpdateTool,
)
from terno_agent.tools.tool_search import TOOL_SEARCH_NAME

SCHEDULE_TOOLS = {
    "schedule_create", "schedule_list", "schedule_get", "schedule_update", "schedule_delete",
}


@pytest.fixture()
def store():
    return InMemoryScheduleStore()


def _weekly(**overrides):
    fields = {
        "name": "Weekly sales",
        "instruction": "Summarise last week's sales by region.",
        "frequency": "weekly",
        "time_of_day": "9:00",
        "timezone": "Asia/Kolkata",
        "day_of_week": 0,
        "recipients": ["Boss@Acme.com", "boss@acme.com", "team@acme.com"],
        "email_subject": "Sales {{run_date}}",
    }
    fields.update(overrides)
    return fields


def test_create_normalises_and_starts_paused(store):
    created = json.loads(ScheduleCreateTool(store).run(**_weekly()))
    assert created["id"] == "1"
    assert created["time_of_day"] == "09:00"
    assert created["recipients"] == ["boss@acme.com", "team@acme.com"]
    assert created["attachment_formats"] == ["xlsx"]
    assert created["is_active"] is False
    assert created["when"] == "Every Monday at 09:00 (Asia/Kolkata)"


def test_create_drops_fields_the_frequency_does_not_use(store):
    created = json.loads(ScheduleCreateTool(store).run(**_weekly(frequency="daily", month=4)))
    assert (created["day_of_week"], created["month"]) == (None, None)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"frequency": "hourly"}, "frequency"),
        ({"time_of_day": "25:00"}, "HH:MM"),
        ({"timezone": "Mars/Base"}, "timezone"),
        ({"day_of_week": None}, "day_of_week"),
        ({"frequency": "quarterly", "day_of_month": 1, "month": 5}, "quarter"),
        ({"recipients": []}, "recipient"),
        ({"recipients": ["not-an-email"]}, "valid email"),
        ({"attachment_formats": ["pdf"]}, "pdf"),
        ({"instruction": "  "}, "instruction"),
    ],
)
def test_create_rejects_bad_input(store, overrides, message):
    with pytest.raises(ToolError, match=message):
        ScheduleCreateTool(store).run(**_weekly(**overrides))
    assert store.list() == []


def test_list_get_update_delete(store):
    ScheduleCreateTool(store).run(**_weekly())

    rows = json.loads(ScheduleListTool(store).run())
    assert [(r["id"], r["name"], r["is_active"]) for r in rows] == [("1", "Weekly sales", False)]

    assert json.loads(ScheduleGetTool(store).run(schedule_id="1"))["day_of_week"] == 0

    updated = json.loads(
        ScheduleUpdateTool(store).run(schedule_id="1", is_active=True, time_of_day="18:30")
    )
    assert (updated["is_active"], updated["time_of_day"], updated["name"]) == (
        True, "18:30", "Weekly sales",
    )

    deleted = json.loads(ScheduleDeleteTool(store).run(schedule_id="1"))
    assert deleted["id"] == "1"
    assert store.list() == []
    with pytest.raises(ToolError, match="Unknown schedule id"):
        ScheduleGetTool(store).run(schedule_id="1")


def test_update_validates_the_merged_schedule(store):
    ScheduleCreateTool(store).run(**_weekly(frequency="daily", day_of_week=None))
    with pytest.raises(ToolError, match="day_of_month"):
        ScheduleUpdateTool(store).run(schedule_id="1", frequency="monthly")
    assert store.get("1").frequency == "daily"
    updated = json.loads(
        ScheduleUpdateTool(store).run(schedule_id="1", frequency="monthly", day_of_month=31)
    )
    assert updated["when"] == "Day 31 of every month at 09:00 (Asia/Kolkata)"


def test_update_needs_a_change(store):
    ScheduleCreateTool(store).run(**_weekly())
    with pytest.raises(ToolError, match="Nothing to update"):
        ScheduleUpdateTool(store).run(schedule_id="1")


class _DummyLLM:
    model = "dummy"

    def complete(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("LLM should not be invoked in this test")


def test_agent_offers_no_schedule_tools_without_a_store():
    agent = TernoAgent(_DummyLLM())
    names = set(agent.tools) | set(agent.deferred_tools)
    assert not names & SCHEDULE_TOOLS
    assert TOOL_SEARCH_NAME not in agent.tools


def test_agent_defers_schedule_tools_behind_tool_search(store):
    agent = TernoAgent(_DummyLLM(), schedule_store=store)
    assert set(agent.deferred_tools) == SCHEDULE_TOOLS
    assert TOOL_SEARCH_NAME in agent.tools
    assert "- schedules: schedule_create" in agent.system_prompt

    agent.tools[TOOL_SEARCH_NAME].run(query="server:schedules")
    assert SCHEDULE_TOOLS <= set(agent.tools)


def test_agent_can_send_schedule_tools_up_front(store):
    agent = TernoAgent(_DummyLLM(), schedule_store=store, defer_schedule_tools=False)
    assert SCHEDULE_TOOLS <= set(agent.tools)
    assert TOOL_SEARCH_NAME not in agent.tools
