import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.database import list_database_identities


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GALLERY_FILE = PROJECT_ROOT / "embeddings" / "gallery_embeddings.npz"

# Exploratory threshold from your previous LFW evaluation.
# We will calibrate this properly later using a separate validation set.
DEFAULT_THRESHOLD = 0.45

_model = None
_gallery_embeddings = None
_gallery_labels = None


def load_recognition_resources():
    """Load InsightFace and the existing embedding gallery."""
    global _model, _gallery_embeddings, _gallery_labels

    if _model is None:
        _model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        _model.prepare(ctx_id=0, det_size=(640, 640))

    if _gallery_embeddings is None:
        if not GALLERY_FILE.exists():
            raise FileNotFoundError(
                f"Gallery file not found: {GALLERY_FILE}"
            )

        with np.load(GALLERY_FILE, allow_pickle=False) as data:
            embeddings = data["embeddings"].astype(np.float32)
            labels = data["labels"].astype(str)

        if embeddings.ndim != 2 or embeddings.shape[1] != 512:
            raise ValueError("Expected gallery embeddings with shape (N, 512).")

        if len(embeddings) != len(labels) or len(labels) == 0:
            raise ValueError("Gallery embeddings and labels are inconsistent.")

        # Normalize gallery vectors for cosine similarity.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Gallery contains a zero-length embedding.")

        _gallery_embeddings = embeddings / norms
        _gallery_labels = labels

    return {
        "model": _model,
        "gallery_size": len(_gallery_labels),
    }




def load_registered_gallery():
    """Load registered face embeddings using SQLite as the source of truth."""
    embeddings = []
    labels = []
    ids = []

    empty_result = (
        np.empty((0, 512), dtype=np.float32),
        np.array([], dtype=str),
        np.array([], dtype=str),
    )

    try:
        people = list_database_identities()
    except Exception as exc:
        raise RuntimeError(
            "Could not load registered identities from SQLite."
        ) from exc

    for person in people:
        embedding_path = PROJECT_ROOT / person["embedding_path"]

        if not embedding_path.is_file():
            continue

        try:
            vector = np.asarray(
                np.load(embedding_path, allow_pickle=False),
                dtype=np.float32,
            ).reshape(-1)
        except (OSError, ValueError):
            continue

        if vector.shape != (512,):
            continue

        norm = np.linalg.norm(vector)

        if not np.isfinite(norm) or norm == 0:
            continue

        embeddings.append(vector / norm)
        labels.append(person["name"])
        ids.append(person["id"])

    if not embeddings:
        return empty_result

    return (
        np.stack(embeddings).astype(np.float32),
        np.asarray(labels, dtype=str),
        np.asarray(ids, dtype=str),
    )




def recognize_image(image, threshold=DEFAULT_THRESHOLD):
    """Recognize faces only against identities registered in FacialIQ."""
    load_recognition_resources()

    if image is None or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")

    faces = _model.get(image)

    if not faces:
        return {
            "status": "no_face",
            "message": "No face detected in the uploaded image.",
        }

    # Preserve the existing largest-face behavior.
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

    # Product recognition uses registered identities only.
    registered_embeddings, registered_labels, registered_ids = (
        load_registered_gallery()
    )

    common = {
        "threshold": threshold,
        "detection_score": round(float(face.det_score), 4),
        "face_count": len(faces),
        "bbox": [round(float(v), 2) for v in face.bbox],
    }

    if len(registered_embeddings) == 0:
        return {
            **common,
            "status": "unknown",
            "identity": None,
            "source": None,
            "registered_id": None,
            "similarity": None,
            "message": "No identities are registered in FacialIQ.",
        }

    scores = registered_embeddings @ embedding
    best_index = int(np.argmax(scores))
    best_similarity = float(scores[best_index])
    matched = best_similarity >= threshold

    return {
        **common,
        "status": "recognized" if matched else "unknown",
        "identity": (
            str(registered_labels[best_index]) if matched else None
        ),
        "source": "registered" if matched else None,
        "registered_id": (
            str(registered_ids[best_index]) if matched else None
        ),
        "similarity": round(best_similarity, 4),
    }



def evaluate_lfw_image(image, threshold=0.45):
    """Evaluate a face against the LFW dataset gallery, not registered users."""
    load_recognition_resources()

    if image is None or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")

    faces = _model.get(image)

    if not faces:
        return {
            "status": "no_face",
            "message": "No face detected in the uploaded image.",
        }

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

    # Use the dataset gallery loaded by load_recognition_resources().
    # These variable names must match the actual gallery variables
    # defined in your existing recognition.py file.
    similarities = _gallery_embeddings @ embedding

    if len(similarities) == 0:
        return {
            "status": "empty_gallery",
            "message": "The LFW gallery contains no embeddings.",
        }

    best_index = int(np.argmax(similarities))
    best_similarity = float(similarities[best_index])
    matched = best_similarity >= threshold

    return {
        "status": "matched" if matched else "unknown",
        "identity": str(_gallery_labels[best_index]) if matched else None,
        "best_candidate": str(_gallery_labels[best_index]),
        "similarity": round(best_similarity, 4),
        "threshold": threshold,
        "detection_score": round(float(face.det_score), 4),
        "face_count": len(faces),
    }



def compare_galleries_image(image, threshold=DEFAULT_THRESHOLD, mode="both"):
    """Compare one detected face against the selected recognition gallery."""
    load_recognition_resources()

    if image is None or image.size == 0:
        raise ValueError("The uploaded image is empty or invalid.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")

    if mode not in {"registered", "lfw", "both"}:
        raise ValueError("Mode must be registered, lfw, or both.")

    faces = _model.get(image)

    if not faces:
        return {
            "status": "no_face",
            "identity": None,
            "source": None,
            "similarity": None,
            "threshold": threshold,
            "face_count": 0,
            "candidates": [],
            "message": "No face detected in the uploaded image.",
        }

    # Keep the existing largest-face behavior.
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0])
        * (f.bbox[3] - f.bbox[1]),
    )

    embedding = np.asarray(face.embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Could not create a valid face embedding.")

    embedding = embedding / norm

    gallery_embeddings = []
    gallery_labels = []
    gallery_sources = []
    gallery_ids = []

    if mode in {"registered", "both"}:
        reg_embeddings, reg_labels, reg_ids = load_registered_gallery()
        if len(reg_embeddings):
            gallery_embeddings.append(reg_embeddings)
            gallery_labels.extend(reg_labels.tolist())
            gallery_sources.extend(["registered"] * len(reg_labels))
            gallery_ids.extend(reg_ids.tolist())

    if mode in {"lfw", "both"}:
        if len(_gallery_embeddings):
            gallery_embeddings.append(_gallery_embeddings)
            gallery_labels.extend(_gallery_labels.tolist())
            gallery_sources.extend(["lfw"] * len(_gallery_labels))
            gallery_ids.extend([None] * len(_gallery_labels))

    common = {
        "mode": mode,
        "threshold": float(threshold),
        "detection_score": round(float(face.det_score), 4),
        "face_count": len(faces),
        "bbox": [round(float(v), 2) for v in face.bbox],
    }

    if not gallery_embeddings:
        return {
            **common,
            "status": "unknown",
            "identity": None,
            "source": None,
            "registered_id": None,
            "similarity": None,
            "candidates": [],
            "message": "The selected gallery contains no usable embeddings.",
        }

    all_embeddings = np.vstack(gallery_embeddings)
    scores = all_embeddings @ embedding
    ranked_indices = np.argsort(scores)[::-1][:5]

    candidates = []
    for index in ranked_indices:
        candidates.append({
            "identity": str(gallery_labels[index]),
            "source": gallery_sources[index],
            "registered_id": gallery_ids[index],
            "similarity": round(float(scores[index]), 4),
            "accepted": bool(scores[index] >= threshold),
        })

    best = candidates[0]
    matched = best["similarity"] >= threshold

    return {
        **common,
        "status": "recognized" if matched else "unknown",
        "identity": best["identity"] if matched else None,
        "source": best["source"] if matched else None,
        "registered_id": (
            best["registered_id"]
            if matched and best["source"] == "registered"
            else None
        ),
        "similarity": best["similarity"],
        "candidates": candidates,
        "message": (
            "A candidate passed the selected threshold."
            if matched
            else "No candidate passed the selected threshold."
        ),
    }
