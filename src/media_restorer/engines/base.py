"""Interface abstraite commune à tous les moteurs de restauration."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from media_restorer.image_io import imread_oriented


class BaseEngine(ABC):
    """Contrat que chaque backend de restauration doit respecter."""

    @abstractmethod
    def restore_array(self, img: np.ndarray) -> np.ndarray:
        """Restaure une image BGR (tableau NumPy) et retourne le résultat."""

    def restore_file(self, input_path: Path, output_path: Path) -> Path:
        """Restaure *input_path* et écrit le résultat dans *output_path*.

        L'image est redressée selon son orientation EXIF au chargement
        (:func:`~media_restorer.image_io.imread_oriented`) ; le fichier écrit
        contient donc les pixels dans le bon sens, sans tag ``Orientation``.
        """
        img = imread_oriented(input_path)
        if img is None:
            raise ValueError(f"Impossible de lire : {input_path}")
        result = self.restore_array(img)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        return output_path.resolve()
