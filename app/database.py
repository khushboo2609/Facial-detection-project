import sqlite3
from pathlib import Path
from contextlib import contextmanager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = PROJECT_ROOT / "storage" / "registered_faces"
DATABASE_DIR = PROJECT_ROOT / "storage" / "database"
DATABASE_FILE = DATABASE_DIR / "facialiq.db"



@contextmanager
def get_connection():
    """Yield a SQLite connection and always close it."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_FILE,
        timeout=10,
    )

    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database():
    """Create the identity table if it does not already exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS identities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                embedding_path TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            )
            """
        )

        # One row per person per day. Rows are kept even if the person is
        # later removed, so attendance history is not lost.
        connection.execute(
            """
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
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_day ON attendance (day)"
        )


def migrate_registry():
    """
    Import existing registry records into SQLite.

    Existing files and registry.json are not modified.
    Existing database records are not overwritten.
    """
    initialize_database()

    registry_file = STORAGE_DIR / "registry.json"

    if not registry_file.exists():
        raise FileNotFoundError(
            f"Existing registry not found: {registry_file}"
        )

    import json

    with registry_file.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    identities = registry.get("identities", [])

    with get_connection() as connection:
        for person in identities:
            person_id = person["id"]
            name = person["name"]
            image_path = person["image_path"]
            embedding_path = person["embedding_path"]

            # Validate existing files before importing a record.
            absolute_image = PROJECT_ROOT / image_path
            absolute_embedding = PROJECT_ROOT / embedding_path

            if not absolute_image.is_file():
                raise FileNotFoundError(
                    f"Image file missing for {person_id}: "
                    f"{absolute_image}"
                )

            if not absolute_embedding.is_file():
                raise FileNotFoundError(
                    f"Embedding file missing for {person_id}: "
                    f"{absolute_embedding}"
                )

            connection.execute(
                """
                INSERT OR IGNORE INTO identities (
                    id, name, image_path, embedding_path
                )
                VALUES (?, ?, ?, ?)
                """,
                (person_id, name, image_path, embedding_path),
            )

    return len(identities)


def list_database_identities():
    """Return identity metadata without exposing it to public APIs."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, name, image_path, embedding_path, created_at
            FROM identities
            ORDER BY created_at, id
            """
        ).fetchall()

    return [dict(row) for row in rows]