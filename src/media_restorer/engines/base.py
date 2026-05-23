"""Interface abstraite commune à tous les moteurs de restauration."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class BaseEngine(ABC):
    """Contrat que chaque backend de restauration doit respecter."""

    @abstractmethod
    def restore_array(self, img: np.ndarray) -> np.ndarray:
        """Restaure une image BGR (tableau NumPy) et retourne le résultat."""

    def restore_file(self, input_path: Path, output_path: Path) -> Path:
        """Restaure *input_path* et écrit le résultat dans *output_path*."""
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Impossible de lire : {input_path}")
        result = self.restore_array(img)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
        return output_path.resolve()
