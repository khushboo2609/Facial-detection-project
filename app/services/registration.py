
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.database import (
    get_connection,
    list_database_identities,
)
from app.services.recognition import load_recognition_resources


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = PROJECT_ROOT / "storage" / "registered_faces"


def _safe_id(name):
    """Create a filesystem-safe identity ID."""
    base = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        name.strip(),
    ).strip("_")

    if not base:
        raise ValueError("Name must contain letters or numbers.")

    return base[:60]


def list_registered():
    """Return identity details without exposing private storage paths."""
    return [
        {
            "id": item["id"],
            "name": item["name"],
        }
        for item in list_database_identities()
    ]


def register_person(name, image):
    """Register a person using SQLite as the identity source of truth."""
    if not name or not name.strip():
        raise ValueError("Name is required.")

    clean_name = name.strip()
    safe_id = _safe_id(clean_name)

    resources = load_recognition_resources()
    model = resources["model"]

    if image is None or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")

    faces = model.get(image)

    if not faces:
        raise ValueError("No face detected. Try a clearer face image.")

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0])
        * (f.bbox[3] - f.bbox[1]),
    )

    embedding = np.asarray(face.embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Could not create a valid face embedding.")

    embedding = embedding / norm

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Use a unique folder ID so different names that sanitize to the
    # same value cannot accidentally overwrite another person's files.
    person_id = safe_id
    person_dir = STORAGE_DIR / person_id

    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id FROM identities WHERE lower(id) = lower(?)",
            (person_id,),
        ).fetchone()

        if existing:
            # Preserve the existing registration and its files.
            raise ValueError("This identity is already registered.")

        if person_dir.exists():
            # Never overwrite files left by an earlier or partial attempt.
            person_id = f"{safe_id}_{uuid.uuid4().hex[:8]}"
            person_dir = STORAGE_DIR / person_id

        image_path = person_dir / "face.jpg"
        embedding_path = person_dir / "embedding.npy"

        person_dir.mkdir(parents=False, exist_ok=False)

        try:
            if not cv2.imwrite(str(image_path), image):
                raise ValueError("Could not save the face image.")

            np.save(embedding_path, embedding)

            relative_image = str(
                image_path.relative_to(PROJECT_ROOT)
            )
            relative_embedding = str(
                embedding_path.relative_to(PROJECT_ROOT)
            )

            connection.execute(
                """
                INSERT INTO identities (
                    id, name, image_path, embedding_path
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    person_id,
                    clean_name,
                    relative_image,
                    relative_embedding,
                ),
            )

        except Exception:
            # Remove files created by this attempt if registration fails.
            for path in (image_path, embedding_path):
                try:
                    if path.exists():
                        path.unlink()
                except OSError:
                    pass

            try:
                person_dir.rmdir()
            except OSError:
                pass

            raise

    return {
        "message": "Identity registered successfully.",
        "id": person_id,
        "name": clean_name,
    }


def delete_person(person_id):
    """
    Delete a registered identity and its files.

    Files are moved to a temporary quarantine directory first.
    The database record is removed only after the file move succeeds.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        person = connection.execute(
            """
            SELECT id, name, image_path, embedding_path
            FROM identities
            WHERE lower(id) = lower(?)
            """,
            (person_id,),
        ).fetchone()

        if person is None:
            raise KeyError("Registered identity not found.")

        actual_id = person["id"]
        person_dir = STORAGE_DIR / actual_id

        # Validate the directory is inside the registered-face storage.
        if person_dir.resolve().parent != STORAGE_DIR.resolve():
            raise ValueError("Invalid registered identity storage path.")

        quarantine_dir = STORAGE_DIR / (
            f".delete_{actual_id}_{uuid.uuid4().hex}"
        )

        moved = False

        try:
            if person_dir.exists():
                person_dir.rename(quarantine_dir)
                moved = True

            connection.execute(
                "DELETE FROM identities WHERE id = ?",
                (actual_id,),
            )

        except Exception:
            # Restore the directory if the database deletion fails.
            if moved and quarantine_dir.exists():
                quarantine_dir.rename(person_dir)
            raise

    # The database transaction is complete. Clean up quarantined files.
    if moved:
        for path in quarantine_dir.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                raise RuntimeError(
                    "Unexpected subdirectory in identity storage; "
                    "manual cleanup is required."
                )
        quarantine_dir.rmdir()

    return {
        "message": "Identity deleted successfully.",
        "id": actual_id,
    }
