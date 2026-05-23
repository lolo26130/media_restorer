"""Film/video restoration pipeline (placeholder — à implémenter).

Stratégie prévue :
- extraire les frames avec OpenCV ou ffmpeg
- appliquer restore_image_array() sur chaque frame
- réassembler en vidéo avec ffmpeg
"""

from __future__ import annotations

from pathlib import Path


def restore_film(
    input_path: Path | str,
    output_path: Path | str,
    model_path: Path | str | None = None,
) -> Path:
    """Restore an old film frame by frame using Real-ESRGAN.

    .. note::
        Not yet implemented.  The function signature is defined so that
        the CLI and future code can reference it without changes.

    Parameters
    ----------
    input_path:
        Path to the source video file.
    output_path:
        Destination path for the restored video.
    model_path:
        Path to the Real-ESRGAN weights file.

    Returns
    -------
    Path
        Absolute path to the restored video.

    Raises
    ------
    NotImplementedError
        Always — film restoration is not yet implemented.
    """
    raise NotImplementedError("Film restoration is not yet implemented.")
