"""Comparison layer: one result format for the original DA3 and all enhancements (see schema.py)."""

from .schema import SCHEMA, load_records, make_record, protocol, protocol_key, save_record, validate
from .tables import extra_markdown, extra_table, sec6_markdown, sec6_table, summary_markdown

__all__ = [
    "SCHEMA", "load_records", "make_record", "protocol", "protocol_key", "save_record", "validate",
    "extra_markdown", "extra_table", "sec6_markdown", "sec6_table", "summary_markdown",
]
