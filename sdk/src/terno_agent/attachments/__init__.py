"""Native attachment ingestion and prompt assembly."""

from terno_agent.attachments.manager import (
    AttachmentInput,
    AttachmentManager,
    AttachmentPolicy,
)
from terno_agent.attachments.store import AttachmentRecord, AttachmentStore

__all__ = [
    "AttachmentInput",
    "AttachmentManager",
    "AttachmentPolicy",
    "AttachmentRecord",
    "AttachmentStore",
]
