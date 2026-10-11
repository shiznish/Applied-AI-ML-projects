"""Audit trail: one immutable record per query, keyed by the per-request
correlation id (app/middleware/correlation_id.py) so a single request stays
traceable across logs, audit records, and OTel traces."""
