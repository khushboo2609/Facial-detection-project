import logging
from contextlib import asynccontextmanager
from app.database import initialize_database
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.services.attendance import (
    ATTENDANCE_MIN_SIMILARITY,
    export_attendance_csv,
    list_attendance,
    mark_from_result,
)

from app.services.recognition import (
    DEFAULT_THRESHOLD,
    load_recognition_resources,
    recognize_image,
    evaluate_lfw_image,
    compare_galleries_image,
)



from app.services.registration import (
    register_person,
    list_registered,
    delete_person,
)


logger = logging.getLogger("facialiq")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the database and load recognition resources at startup."""
    initialize_database()
    load_recognition_resources()
    yield


app = FastAPI(
    title="ArcFace Facial Recognition API",
    summary="Face Registration, Recognition and Identity Management",
    description="""
# ArcFace Facial Recognition System

Welcome to the API documentation for the ArcFace face recognition project.

## Main Features
- Face recognition using ArcFace embeddings
- Face registration and identity management
- Face similarity matching
- Unknown-person detection
- Health monitoring

## How to use
1. Check the API health.
2. Register a person's face.
3. Upload an image for recognition.
4. View or delete registered identities.
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "System",
            "description": "API health checks and system information."
        },
        {
            "name": "Face Recognition",
            "description": "Upload images and identify faces using ArcFace."
        },
        {
            "name": "Identity Management",
            "description": "Register new people, list registered identities, and delete registrations."
        },
        {
            "name": "Attendance",
            "description": "Daily first-seen and last-seen times for registered people, with CSV export."
        }
    ]
)

# Development setting for a separately running frontend.
# Restrict this to your frontend origin before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root():
    return {
        "message": "ArcFace Face Recognition API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["System"])
def health():
    resources = load_recognition_resources()
    return {
        "status": "ok",
        "model": "buffalo_l",
        "provider": "CPUExecutionProvider",
        "gallery_size": resources["gallery_size"],
    }


@app.post("/recognize", tags=["Face Recognition"])
async def recognize(
    file: UploadFile = File(...),
    threshold: float = Query(
        DEFAULT_THRESHOLD, ge=-1.0, le=1.0
    ),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Basic upload-size protection: 10 MB.
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit.",
        )

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image.",
        )

    try:
        return recognize_image(image, threshold=threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/register", tags=["Identity Management"])
async def register(
    name: str = Query(..., min_length=1, max_length=100),
    file: UploadFile = File(...),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB.")

    image = cv2.imdecode(
        np.frombuffer(contents, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    try:
        return register_person(name, image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/registered", tags=["Identity Management"])
def get_registered():
    return {"identities": list_registered()}


@app.delete("/registered/{person_id}", tags=["Identity Management"])
def remove_registered(person_id: str):
    try:
        return delete_person(person_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/evaluate-lfw")
async def evaluate_lfw(
    file: UploadFile = File(...),
    threshold: float = Query(0.45, ge=-1.0, le=1.0),
):
    contents = await file.read()

    image = cv2.imdecode(
        np.frombuffer(contents, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    try:
        return evaluate_lfw_image(image, threshold)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.post("/compare-galleries", tags=["Face Recognition"])
async def compare_galleries(
    file: UploadFile = File(...),
    mode: str = Query("both", pattern="^(registered|lfw|both)$"),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0.0, le=1.0),
    mark_attendance: bool = Query(
        False,
        description=(
            "Also record today's attendance when the best match is a "
            f"registered person scoring at least {ATTENDANCE_MIN_SIMILARITY}."
        ),
    ),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit.",
        )

    image = cv2.imdecode(
        np.frombuffer(contents, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded image.",
        )

    try:
        result = compare_galleries_image(
            image,
            threshold=threshold,
            mode=mode,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if mark_attendance:
        # Attendance must never break recognition. If saving fails, the
        # recognition result is still returned and the error is reported.
        try:
            result["attendance"] = mark_from_result(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not mark attendance")
            result["attendance"] = {
                "status": "error",
                "message": f"Attendance could not be saved: {exc}",
            }

    return result


_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


@app.get("/attendance", tags=["Attendance"])
def get_attendance(day: str | None = Query(None, pattern=_DATE_PATTERN)):
    """Who was present and who was absent on a day (default: today)."""
    try:
        return list_attendance(day)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/attendance/export", tags=["Attendance"])
def export_attendance(
    start: str | None = Query(None, pattern=_DATE_PATTERN),
    end: str | None = Query(None, pattern=_DATE_PATTERN),
):
    """Download attendance as a CSV file. No dates means every day."""
    try:
        content = export_attendance_csv(start, end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if start and end and start == end:
        label = start
    else:
        label = f"{start or 'start'}_to_{end or 'latest'}"

    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="attendance_{label}.csv"'
        },
    )