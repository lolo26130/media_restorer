"""Persistance HDF5 des tracés (:class:`~media_restorer.engines.vectorise.types.StrokeSet`).

Schéma du fichier ::

    /meta                          (attrs: source_path, image_shape, dpi, created_at)
    /candidates/<i>                (attrs: label, score, pencil_width_px)
    /candidates/<i>/strokes/<j>/points      (N,2) float32
    /candidates/<i>/strokes/<j>/widths      (N,)  float32
    /candidates/<i>/strokes/<j>/intensity   (N,)  float32
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py

from media_restorer.engines.vectorise.types import Stroke, StrokeSet


def save_strokes(
    path: Path,
    candidates: list[StrokeSet],
    *,
    source_path: Path | str,
    dpi: float,
    image_shape: tuple[int, int],
) -> None:
    """Écrit *candidates* dans *path* (HDF5) — voir docstring de module pour le schéma."""
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["source_path"] = str(source_path)
        meta.attrs["image_shape"] = image_shape
        meta.attrs["dpi"] = dpi
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()

        cand_group = f.create_group("candidates")
        for i, stroke_set in enumerate(candidates):
            g = cand_group.create_group(str(i))
            g.attrs["label"] = stroke_set.label
            g.attrs["score"] = stroke_set.score
            g.attrs["pencil_width_px"] = stroke_set.pencil_width_px
            strokes_group = g.create_group("strokes")
            for j, stroke in enumerate(stroke_set.strokes):
                sg = strokes_group.create_group(str(j))
                sg.create_dataset("points", data=stroke.points, compression="gzip")
                sg.create_dataset("widths", data=stroke.widths, compression="gzip")
                sg.create_dataset("intensity", data=stroke.intensity, compression="gzip")


def load_strokes(path: Path) -> list[StrokeSet]:
    """Relit un fichier écrit par :func:`save_strokes`.

    Fait partie de l'API du cœur de calcul (testable) mais n'est pas encore
    câblée à une action de l'extension Qt — voir la section « Hors
    périmètre » du plan d'implémentation : un bouton « charger des tracés »
    est immédiat à ajouter le jour où il est demandé.
    """
    result: list[StrokeSet] = []
    with h5py.File(path, "r") as f:
        cand_group = f["candidates"]
        for key in sorted(cand_group.keys(), key=int):
            g = cand_group[key]
            strokes_group = g["strokes"]
            strokes = [
                Stroke(
                    points=strokes_group[skey]["points"][()],
                    widths=strokes_group[skey]["widths"][()],
                    intensity=strokes_group[skey]["intensity"][()],
                )
                for skey in sorted(strokes_group.keys(), key=int)
            ]
            result.append(StrokeSet(
                strokes=strokes,
                pencil_width_px=float(g.attrs["pencil_width_px"]),
                score=float(g.attrs["score"]),
                label=str(g.attrs["label"]),
            ))
    return result
