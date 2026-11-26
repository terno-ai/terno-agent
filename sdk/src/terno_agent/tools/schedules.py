"""Scheduling tools and the canonical schedule-store contract.

A schedule asks the agent to carry out an instruction on a fixed preset
(daily, weekly, monthly, quarterly or yearly at a time of day in a timezone)
and email the result. This module defines:

* :class:`Schedule` — the neutral schedule record shared everywhere.
* :class:`ScheduleStore` — an abstract base that owns field validation and
  normalisation. Concrete stores only implement five persistence primitives
  (:meth:`~ScheduleStore._create`, :meth:`~ScheduleStore._apply_update`,
  :meth:`~ScheduleStore._delete`, :meth:`~ScheduleStore.get`,
  :meth:`~ScheduleStore.list`), so integrations never re-implement the rules.
* :class:`InMemoryScheduleStore` — a process-local store for standalone use
  and tests. It only records schedules; nothing runs them.
* The five LLM-facing tools: ``schedule_create``, ``schedule_list``,
  ``schedule_get``, ``schedule_update``, ``schedule_delete``.

Hosts that actually run schedules (e.g. terno-ai's database-backed store)
subclass :class:`ScheduleStore`, may add their own checks (permissions,
allowed email domains, datasource access) by raising :class:`ToolError`
from the primitives, and fill in ``next_run_at`` and ``url``.

The tools are only offered when a host passes a store, and they are
deferred behind ``tool_search`` by default, so a chat that never schedules
anything pays only for their names.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from terno_agent.core.exceptions import ToolError
from terno_agent.core.tool import ToolSchema

FREQUENCIES = ("daily", "weekly", "monthly", "quarterly", "yearly")
ATTACHMENT_FORMATS = ("xlsx", "csv", "chart")
_USES_DAY_OF_MONTH = ("monthly", "quarterly", "yearly")
_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

NAME_MAX_CHARS = 255
SUBJECT_MAX_CHARS = 255
INSTRUCTION_MAX_CHARS = 8000

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Fields a caller may set; everything else on Schedule is owned by the store.
_EDITABLE = (
    "name", "instruction", "frequency", "time_of_day", "timezone",
    "day_of_week", "day_of_month", "month", "recipients", "email_subject",
    "email_body", "attachment_formats", "datasource_id", "is_active",
)


@dataclass
class Schedule:
    id: str
    name: str
    instruction: str
    frequency: str
    time_of_day: str  # "HH:MM", in ``timezone``
    timezone: str = "UTC"
    day_of_week: int | None = None  # weekly: 0 = Monday … 6 = Sunday
    day_of_month: int | None = None  # monthly/quarterly/yearly: 1–31 (clamped to month end)
    month: int | None = None  # yearly: 1–12; quarterly: 1–3 (month of the quarter)
    recipients: list[str] = field(default_factory=list)
    email_subject: str = ""
    email_body: str = ""
    attachment_formats: list[str] = field(default_factory=lambda: ["xlsx"])
    datasource_id: int | None = None
    is_active: bool = False
    # Filled in by the host; the in-memory store leaves them empty.
    next_run_at: str | None = None
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["when"] = describe_when(self)
        return data


def describe_when(schedule: Schedule) -> str:
    """Plain-English recurrence, e.g. 'Every Monday at 09:00 (Asia/Kolkata)'."""
    at = f"at {schedule.time_of_day} ({schedule.timezone})"
    freq = schedule.frequency
    if freq == "daily":
        return f"Every day {at}"
    if freq == "weekly" and schedule.day_of_week is not None:
        return f"Every {_DAY_NAMES[schedule.day_of_week]} {at}"
    if freq == "monthly":
        return f"Day {schedule.day_of_month} of every month {at}"
    if freq == "quarterly":
        return f"Day {schedule.day_of_month} of month {schedule.month} of every quarter {at}"
    if freq == "yearly":
        return f"Every year on day {schedule.day_of_month} of month {schedule.month} {at}"
    return f"{freq} {at}"


# --------------------------------------------------------------------------- #
# Validation (shared by every store)
# --------------------------------------------------------------------------- #


def _text(fields: dict[str, Any], key: str, *, required: bool, limit: int, label: str) -> str:
    value = str(fields.get(key) or "").strip()
    if required and not value:
        raise ToolError(f"{label} is required.")
    if len(value) > limit:
        raise ToolError(f"{label} must be {limit} characters or fewer.")
    return value


def _int_or_none(value: Any, label: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{label} must be a whole number.") from exc


def _recipients(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = re.split(r"[,;\s]+", raw)
    if not isinstance(raw, (list, tuple)):
        raise ToolError("recipients must be a list of email addresses.")
    out: list[str] = []
    for value in raw:
        email = str(value or "").strip().lower()
        if not email or email in out:
            continue
        if not _EMAIL_RE.match(email):
            raise ToolError(f"{email} is not a valid email address.")
        out.append(email)
    if not out:
        raise ToolError("Add at least one recipient.")
    return out


def _formats(raw: Any) -> list[str]:
    if raw is None:
        return ["xlsx"]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ToolError("attachment_formats must be a list.")
    out: list[str] = []
    for value in raw:
        fmt = str(value).strip().lower()
        if fmt not in ATTACHMENT_FORMATS:
            raise ToolError(
                f"Unsupported attachment format {fmt!r}. Use: {', '.join(ATTACHMENT_FORMATS)}."
            )
        if fmt not in out:
            out.append(fmt)
    if not out:
        raise ToolError("Choose at least one attachment format.")
    return out


def validate_schedule_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Check and normalise a full set of schedule fields; raise ToolError on the first problem.

    Recurrence fields that the frequency doesn't use are cleared, so a
    schedule switched from weekly to daily drops its day of the week.
    """
    frequency = str(fields.get("frequency") or "").strip().lower()
    if frequency not in FREQUENCIES:
        raise ToolError(f"frequency must be one of: {', '.join(FREQUENCIES)}.")

    time_of_day = str(fields.get("time_of_day") or "").strip()
    match = _TIME_RE.match(time_of_day)
    if not match:
        raise ToolError("time_of_day must be 24-hour HH:MM, e.g. '09:00'.")
    time_of_day = f"{int(match.group(1)):02d}:{match.group(2)}"

    timezone = str(fields.get("timezone") or "UTC").strip()
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ToolError(
            f"Unknown timezone {timezone!r}; use an IANA name like 'Asia/Kolkata'."
        ) from exc

    day_of_week = _int_or_none(fields.get("day_of_week"), "day_of_week")
    day_of_month = _int_or_none(fields.get("day_of_month"), "day_of_month")
    month = _int_or_none(fields.get("month"), "month")
    if frequency == "weekly" and (day_of_week is None or not 0 <= day_of_week <= 6):
        raise ToolError("A weekly schedule needs day_of_week (0 = Monday … 6 = Sunday).")
    if frequency in _USES_DAY_OF_MONTH and (day_of_month is None or not 1 <= day_of_month <= 31):
        raise ToolError(f"A {frequency} schedule needs day_of_month between 1 and 31.")
    if frequency == "quarterly" and (month is None or not 1 <= month <= 3):
        raise ToolError(
            "A quarterly schedule needs month = 1, 2 or 3 (which month of each quarter)."
        )
    if frequency == "yearly" and (month is None or not 1 <= month <= 12):
        raise ToolError("A yearly schedule needs month between 1 and 12.")

    return {
        "name": _text(fields, "name", required=True, limit=NAME_MAX_CHARS, label="name"),
        "instruction": _text(
            fields, "instruction", required=True, limit=INSTRUCTION_MAX_CHARS, label="instruction"
        ),
        "frequency": frequency,
        "time_of_day": time_of_day,
        "timezone": timezone,
        "day_of_week": day_of_week if frequency == "weekly" else None,
        "day_of_month": day_of_month if frequency in _USES_DAY_OF_MONTH else None,
        "month": month if frequency in ("quarterly", "yearly") else None,
        "recipients": _recipients(fields.get("recipients")),
        "email_subject": _text(
            fields, "email_subject", required=True, limit=SUBJECT_MAX_CHARS, label="email_subject"
        ),
        "email_body": str(fields.get("email_body") or "").strip(),
        "attachment_formats": _formats(fields.get("attachment_formats")),
        "datasource_id": _int_or_none(fields.get("datasource_id"), "datasource_id"),
        "is_active": bool(fields.get("is_active", False)),
    }


# --------------------------------------------------------------------------- #
# Stores
# --------------------------------------------------------------------------- #


class ScheduleStore(ABC):
    """Canonical schedule-store contract shared by terno-agent and terno-ai.

    Subclasses implement only the persistence primitives. This base owns
    validation, so every store accepts and rejects the same input:

    * :meth:`create` validates the full field set; new schedules start
      paused unless ``is_active`` is passed.
    * :meth:`update` merges the changes onto the stored schedule and
      validates the result as a whole (switching to ``weekly`` needs a
      ``day_of_week`` even if only ``frequency`` was sent).
    """

    def create(self, **fields: Any) -> Schedule:
        clean = validate_schedule_fields(fields)
        return self._create(clean)

    def update(self, schedule_id: str, **changes: Any) -> Schedule:
        changes = {k: v for k, v in changes.items() if k in _EDITABLE and v is not None}
        if not changes:
            raise ToolError("Nothing to update: pass at least one field to change.")
        current = self.get(str(schedule_id))
        merged = {key: getattr(current, key) for key in _EDITABLE}
        merged.update(changes)
        clean = validate_schedule_fields(merged)
        return self._apply_update(str(schedule_id), clean)

    def delete(self, schedule_id: str) -> Schedule:
        schedule = self.get(str(schedule_id))
        self._delete(str(schedule_id))
        return schedule

    # ----- persistence primitives (subclass implements) ---------------- #

    @abstractmethod
    def _create(self, fields: dict[str, Any]) -> Schedule:
        """Persist a new schedule from already-validated ``fields`` and return it."""

    @abstractmethod
    def _apply_update(self, schedule_id: str, fields: dict[str, Any]) -> Schedule:
        """Overwrite the stored schedule with already-validated ``fields`` and return it."""

    @abstractmethod
    def _delete(self, schedule_id: str) -> None:
        """Remove the schedule; :meth:`get` has already confirmed it exists."""

    @abstractmethod
    def get(self, schedule_id: str) -> Schedule:
        """Return one schedule by id, or raise :class:`ToolError`."""

    @abstractmethod
    def list(self) -> list[Schedule]:
        """Return the caller's schedules, oldest first."""


class InMemoryScheduleStore(ScheduleStore):
    """Thread-safe, process-local schedule list. Records schedules; never runs them."""

    def __init__(self) -> None:
        self._schedules: dict[str, Schedule] = {}
        self._next_id = 1
        self._lock = Lock()

    def _create(self, fields: dict[str, Any]) -> Schedule:
        with self._lock:
            schedule_id = str(self._next_id)
            self._next_id += 1
            schedule = Schedule(id=schedule_id, **fields)
            self._schedules[schedule_id] = schedule
            return schedule

    def _apply_update(self, schedule_id: str, fields: dict[str, Any]) -> Schedule:
        with self._lock:
            schedule = self.get(schedule_id)
            for key, value in fields.items():
                setattr(schedule, key, value)
            return schedule

    def _delete(self, schedule_id: str) -> None:
        with self._lock:
            self._schedules.pop(schedule_id, None)

    def get(self, schedule_id: str) -> Schedule:
        try:
            return self._schedules[str(schedule_id)]
        except KeyError as exc:
            raise ToolError(f"Unknown schedule id: {schedule_id}") from exc

    def list(self) -> list[Schedule]:
        return list(self._schedules.values())


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

# Groups the tools as "schedules" in the deferred-tools index, so
# ``tool_search`` can load them all with ``server:schedules``.
_SERVER = "schedules"

_FIELD_PROPS: dict[str, Any] = {
    "name": {"type": "string", "description": "Short name, e.g. 'Weekly sales digest'."},
    "instruction": {
        "type": "string",
        "description": (
            "What to do on every run, written as a self-contained request: the run "
            "has no chat history, so include every detail (metrics, filters, period "
            "such as 'the previous 7 days', breakdowns, chart types)."
        ),
    },
    "frequency": {"type": "string", "enum": list(FREQUENCIES)},
    "time_of_day": {"type": "string", "description": "24-hour local time, HH:MM, e.g. '09:00'."},
    "timezone": {
        "type": "string",
        "description": "IANA timezone the time is in, e.g. 'Asia/Kolkata'. Ask if unknown.",
    },
    "day_of_week": {
        "type": "integer",
        "minimum": 0,
        "maximum": 6,
        "description": "weekly only: 0 = Monday … 6 = Sunday.",
    },
    "day_of_month": {
        "type": "integer",
        "minimum": 1,
        "maximum": 31,
        "description": (
            "monthly/quarterly/yearly: 1–31; days past the month's end run on its last day."
        ),
    },
    "month": {
        "type": "integer",
        "minimum": 1,
        "maximum": 12,
        "description": "yearly: 1–12. quarterly: 1–3, i.e. which month of each quarter.",
    },
    "recipients": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Email addresses that receive the result.",
    },
    "email_subject": {
        "type": "string",
        "description": "Email subject. May use {{app_name}} and {{run_date}}.",
    },
    "email_body": {
        "type": "string",
        "description": "Optional note shown above the result in the email.",
    },
    "attachment_formats": {
        "type": "array",
        "items": {"type": "string", "enum": list(ATTACHMENT_FORMATS)},
        "description": (
            "Files to attach: xlsx (Excel), csv, chart (interactive HTML). Default ['xlsx']."
        ),
    },
    "datasource_id": {
        "type": "integer",
        "description": "Optional id of the datasource to use, when the instruction is about one.",
    },
}

_ID_PROP = {"schedule_id": {"type": "string", "description": "The schedule's id."}}


def _dump(schedule: Schedule) -> str:
    return json.dumps(schedule.to_dict())


@dataclass
class ScheduleCreateTool:
    store: ScheduleStore
    server = _SERVER

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="schedule_create",
            description=(
                "Create a recurring task: on every run the agent carries out the "
                "instruction from scratch and emails the answer, with the requested "
                "files attached, to the recipients. Use it when the user wants "
                "something done or sent regularly ('every Monday send me…', 'email "
                "the team a monthly report…').\n\n"
                "Before calling:\n"
                "- Collect everything required. Ask the user (ask_user when available) "
                "for anything missing: what to produce, how often and when, timezone, "
                "recipients, and attachments. Don't invent recipients.\n"
                "- When practical, do the task once now so the user can check the "
                "output, then turn it into the instruction.\n"
                "- Read the summary back (what, when, who) and get a yes.\n\n"
                "The schedule is created PAUSED unless is_active is true. Set "
                "is_active only if the user explicitly said to start it now. "
                "Afterwards tell the user the schedule's name, `when`, recipients "
                "and whether it is active. Returns the schedule as JSON."
            ),
            parameters={
                "type": "object",
                "properties": {
                    **_FIELD_PROPS,
                    "is_active": {
                        "type": "boolean",
                        "description": "Start running right away. Default false (paused).",
                    },
                },
                "required": [
                    "name", "instruction", "frequency", "time_of_day", "timezone",
                    "recipients", "email_subject",
                ],
            },
        )

    def run(self, **kwargs: Any) -> str:
        return _dump(self.store.create(**kwargs))


@dataclass
class ScheduleListTool:
    store: ScheduleStore
    server = _SERVER

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="schedule_list",
            description=(
                "List the user's scheduled tasks with their id, name, when they run, "
                "recipients and whether they are active. Use it to find a schedule's "
                "id before schedule_get, schedule_update or schedule_delete. "
                "Returns a JSON array."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def run(self, **_kwargs: Any) -> str:
        rows = [
            {
                "id": s.id,
                "name": s.name,
                "when": describe_when(s),
                "recipients": s.recipients,
                "is_active": s.is_active,
                "next_run_at": s.next_run_at,
            }
            for s in self.store.list()
        ]
        return json.dumps(rows)


@dataclass
class ScheduleGetTool:
    store: ScheduleStore
    server = _SERVER

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="schedule_get",
            description="Fetch one scheduled task in full by id. Returns the schedule as JSON.",
            parameters={
                "type": "object",
                "properties": dict(_ID_PROP),
                "required": ["schedule_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        schedule_id = kwargs.get("schedule_id")
        if not schedule_id:
            raise ToolError("schedule_get requires a 'schedule_id'.")
        return _dump(self.store.get(str(schedule_id)))


@dataclass
class ScheduleUpdateTool:
    store: ScheduleStore
    server = _SERVER

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="schedule_update",
            description=(
                "Change a scheduled task. Pass only the fields to change; the rest "
                "are kept. Pause with is_active=false and resume with is_active=true "
                "(only when the user asks). Changing frequency may need the matching "
                "day_of_week / day_of_month / month. Returns the updated schedule as JSON."
            ),
            parameters={
                "type": "object",
                "properties": {
                    **_ID_PROP,
                    **_FIELD_PROPS,
                    "is_active": {
                        "type": "boolean",
                        "description": "false = pause, true = resume.",
                    },
                },
                "required": ["schedule_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        schedule_id = kwargs.pop("schedule_id", None)
        if not schedule_id:
            raise ToolError("schedule_update requires a 'schedule_id'.")
        return _dump(self.store.update(str(schedule_id), **kwargs))


@dataclass
class ScheduleDeleteTool:
    store: ScheduleStore
    server = _SERVER

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="schedule_delete",
            description=(
                "Delete a scheduled task and its run history. This cannot be undone: "
                "confirm with the user first, naming the schedule. To stop it "
                "temporarily, use schedule_update with is_active=false instead. "
                "Returns the deleted schedule as JSON."
            ),
            parameters={
                "type": "object",
                "properties": dict(_ID_PROP),
                "required": ["schedule_id"],
            },
        )

    def run(self, **kwargs: Any) -> str:
        schedule_id = kwargs.get("schedule_id")
        if not schedule_id:
            raise ToolError("schedule_delete requires a 'schedule_id'.")
        return _dump(self.store.delete(str(schedule_id)))


def schedule_tools(store: ScheduleStore) -> list:
    """All five schedule tools bound to ``store``."""
    return [
        ScheduleCreateTool(store),
        ScheduleListTool(store),
        ScheduleGetTool(store),
        ScheduleUpdateTool(store),
        ScheduleDeleteTool(store),
    ]


__all__ = [
    "ATTACHMENT_FORMATS",
    "FREQUENCIES",
    "InMemoryScheduleStore",
    "Schedule",
    "ScheduleCreateTool",
    "ScheduleDeleteTool",
    "ScheduleGetTool",
    "ScheduleListTool",
    "ScheduleStore",
    "ScheduleUpdateTool",
    "describe_when",
    "schedule_tools",
    "validate_schedule_fields",
]
