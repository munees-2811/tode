"""Frame browsing and image serving endpoints."""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from server.schemas.annotation import FrameAnnotationsIn, FrameAnnotationsOut, FrameOut
from server.services.annotation_service import (
    get_frame_annotations,
    list_frames,
    save_frame_annotations,
)
from server.services.project_service import ProjectService

router = APIRouter(prefix="/api/projects/{project_id}/frames")
_svc   = ProjectService()

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _frame_file(project_id: str, frame_index: int) -> str | None:
    frames_dir = _svc.frames_dir(project_id)
    for ext in _IMG_EXTS:
        p = os.path.join(frames_dir, f"frame_{frame_index:06d}{ext}")
        if os.path.isfile(p):
            return p
    return None


@router.get("", response_model=list[FrameOut])
def get_frames(project_id: str):
    if _svc.get(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return list_frames(project_id)


@router.get("/{frame_index}/image")
def get_frame_image(project_id: str, frame_index: int):
    path = _frame_file(project_id, frame_index)
    if not path:
        raise HTTPException(status_code=404, detail="Frame image not found")
    return FileResponse(path)


@router.get("/{frame_index}/annotations", response_model=FrameAnnotationsOut)
def get_annotations(project_id: str, frame_index: int):
    if _svc.get(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return get_frame_annotations(project_id, frame_index)


@router.post("/{frame_index}/annotations", status_code=204)
def save_annotations(project_id: str, frame_index: int, body: FrameAnnotationsIn):
    if _svc.get(project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    save_frame_annotations(project_id, frame_index, body)
