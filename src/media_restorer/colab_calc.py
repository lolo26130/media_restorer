"""Mixin ColabCalc : délègue le traitement d'images à un serveur FastAPI
hébergé dans Google Colab et exposé via un tunnel cloudflared.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class _ColabRestoreWorker(QThread):
    """Envoie une image au serveur Colab et récupère le résultat (BGR numpy)."""

    result_ready = pyqtSignal(np.ndarray)
    error        = pyqtSignal(str)

    def __init__(self, url: str, img: np.ndarray, engine_name: str) -> None:
        super().__init__()
        self._url         = url
        self._img         = img
        self._engine_name = engine_name

    def run(self) -> None:
        import requests  # import différé — non requis si Colab non utilisé

        ok, buf = cv2.imencode(".png", self._img)
        if not ok:
            self.error.emit("Impossible d'encoder l'image en PNG")
            return
        try:
            resp = requests.post(
                f"{self._url}/process",
                files={"file": ("input.png", buf.tobytes(), "image/png")},
                data={"engine": self._engine_name},
                timeout=180,
            )
            resp.raise_for_status()
            raw    = base64.b64decode(resp.json()["image"])
            arr    = np.frombuffer(raw, dtype=np.uint8)
            result = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if result is None:
                self.error.emit("Réponse Colab invalide (image corrompue)")
                return
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class ColabCalc:
    """Mixin : traitement d'images via Google Colab (FastAPI + tunnel cloudflared).

    L'URL du tunnel est saisie une fois par session via ``_colab__set_url``.
    Les attributs d'état sont name-mangés (``__url`` → ``_ColabCalc__url``) pour
    les isoler du reste de la fenêtre principale qui hérite de cette classe.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__url: str | None                        = None
        self._colab_worker: _ColabRestoreWorker | None = None

    # ------------------------------------------------------------------
    # API privée (name-mangled)
    # ------------------------------------------------------------------

    def _colab__set_url(self, url: str) -> None:
        self.__url = url.rstrip("/")

    def _colab__get_url(self) -> str | None:
        return self.__url

    def _colab__is_connected(self) -> bool:
        """GET /health → True si le serveur Colab répond."""
        if not self.__url:
            return False
        try:
            import requests
            return requests.get(f"{self.__url}/health", timeout=4).ok
        except Exception:
            return False

    def _colab__start_restore(
        self,
        img: np.ndarray,
        engine_name: str,
        on_done,
        on_error,
    ) -> None:
        """Lance _ColabRestoreWorker. *on_done* / *on_error* sont des slots Qt."""
        if not self.__url:
            on_error("URL Colab non configurée — Colab › Connecter.")
            return
        self._colab_worker = _ColabRestoreWorker(self.__url, img, engine_name)
        self._colab_worker.result_ready.connect(on_done)
        self._colab_worker.error.connect(on_error)
        self._colab_worker.start()
