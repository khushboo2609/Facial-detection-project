"""Daily attendance: the first and last time each registered person was seen.

One row per person per day. Only identities registered in FacialIQ can mark
attendance. Dataset (LFW) matches and unknown faces are never logged.
Times are stored in the server computer's local time.
"""

import csv
import io
from datetime import date, datetime

from app.database import get_connection


# Seen again after at least this many seconds counts as a new "visit".
VISIT_GAP_SECONDS = 300

# A face must score at least this similarity to mark attendance. This is
# deliberately stricter than the default recognition threshold (0.45) so a
# borderline match cannot mark the wrong person present.
ATTENDANCE_MIN_SIMILARITY = 0.50


_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS attendance (
        person_id TEXT NOT NULL,
        person_name TEXT NOT NULL,
        day TEXT NOT NULL,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        visits INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (person_id, day)
    )
"""


def _ensure_table(connection):
    """Create the attendance table if it does not exist yet.

    This keeps attendance working even if the database was created before
    the attendance feature was added.
    """
    connection.execute(_TABLE_SQL)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_attendance_day ON attendance (day)"
    )


def _now():
    return datetime.now().replace(microsecond=0)


def _clean_day(value):
    """Return a YYYY-MM-DD string, or raise ValueError."""
    if value is None:
        return date.today().isoformat()
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError("Date must look like 2026-11-12.") from exc


def log_sighting(person_id, now=None):
    """Record that a registered person was seen now.

    Returns a dict whose status is "checked_in" (first time today)
    or "updated" (already marked today; last_seen refreshed).
    """
    now = now or _now()
    day = now.date().isoformat()
    stamp = now.isoformat(timespec="seconds")

    with get_connection() as connection:
        _ensure_table(connection)

        person = connection.execute(
            "SELECT id, name FROM identities WHERE id = ?",
            (person_id,),
        ).fetchone()

        if person is None:
            raise KeyError("Registered identity not found.")

        row = connection.execute(
            """
            SELECT first_seen, last_seen, visits
            FROM attendance
            WHERE person_id = ? AND day = ?
            """,
            (person["id"], day),
        ).fetchone()

        if row is None:
            connection.execute(
                """
                INSERT INTO attendance
                    (person_id, person_name, day, first_seen, last_seen, visits)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (person["id"], person["name"], day, stamp, stamp),
            )
            return {
                "status": "checked_in",
                "person_id": person["id"],
                "name": person["name"],
                "day": day,
                "first_seen": stamp,
                "last_seen": stamp,
                "visits": 1,
            }

        previous = datetime.fromisoformat(row["last_seen"])
        gap = (now - previous).total_seconds()
        visits = row["visits"] + (1 if gap >= VISIT_GAP_SECONDS else 0)
        last_seen = max(stamp, row["last_seen"])

        connection.execute(
            """
            UPDATE attendance
            SET last_seen = ?, visits = ?, person_name = ?
            WHERE person_id = ? AND day = ?
            """,
            (last_seen, visits, person["name"], person["id"], day),
        )

        return {
            "status": "updated",
            "person_id": person["id"],
            "name": person["name"],
            "day": day,
            "first_seen": row["first_seen"],
            "last_seen": last_seen,
            "visits": visits,
        }


def mark_from_result(result):
    """Mark attendance from a /compare-galleries result when it qualifies."""
    if (
        result.get("status") != "recognized"
        or result.get("source") != "registered"
        or not result.get("registered_id")
    ):
        return {"status": "not_applicable"}

    if result["similarity"] < ATTENDANCE_MIN_SIMILARITY:
        return {
            "status": "low_confidence",
            "min_similarity": ATTENDANCE_MIN_SIMILARITY,
        }

    try:
        return log_sighting(result["registered_id"])
    except KeyError:
        return {"status": "not_applicable"}


def list_attendance(day=None):
    """Return who was present on a day and which registered people were not."""
    day = _clean_day(day)

    with get_connection() as connection:
        _ensure_table(connection)

        present_rows = connection.execute(
            """
            SELECT person_id, person_name, first_seen, last_seen, visits
            FROM attendance
            WHERE day = ?
            ORDER BY first_seen, person_name
            """,
            (day,),
        ).fetchall()

        # Only people who were already registered on that day can be absent.
        registered_rows = connection.execute(
            """
            SELECT id, name
            FROM identities
            WHERE date(created_at, 'localtime') <= ?
            ORDER BY name
            """,
            (day,),
        ).fetchall()

    present = [
        {
            "person_id": row["person_id"],
            "name": row["person_name"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
            "visits": row["visits"],
        }
        for row in present_rows
    ]

    present_ids = {row["person_id"] for row in present_rows}
    absent = [
        {"person_id": row["id"], "name": row["name"]}
        for row in registered_rows
        if row["id"] not in present_ids
    ]

    return {
        "day": day,
        "present": present,
        "absent": absent,
        "registered_count": len(registered_rows),
    }


def _safe_cell(value):
    """Stop spreadsheet programs from running a name as a formula."""
    text = str(value)
    if text[:1] in ("=", "+", "-", "@"):
        return "'" + text
    return text


def export_attendance_csv(start=None, end=None):
    """Return attendance rows between two dates (inclusive) as CSV text."""
    start = _clean_day(start) if start else "0000-01-01"
    end = _clean_day(end) if end else "9999-12-31"

    if start > end:
        raise ValueError("Start date must not be after the end date.")

    with get_connection() as connection:
        _ensure_table(connection)

        rows = connection.execute(
            """
            SELECT day, person_name, person_id, first_seen, last_seen, visits
            FROM attendance
            WHERE day BETWEEN ? AND ?
            ORDER BY day, person_name
            """,
            (start, end),
        ).fetchall()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["date", "name", "person_id", "first_seen", "last_seen", "visits"]
    )
    for row in rows:
        writer.writerow(
            [
                row["day"],
                _safe_cell(row["person_name"]),
                _safe_cell(row["person_id"]),
                row["first_seen"][11:19],
                row["last_seen"][11:19],
                row["visits"],
            ]
        )

    # The BOM makes Excel read names with accents correctly.
    return "﻿" + buffer.getvalue()