"""Détection zero-shot de repères nommés sur une image (dessins/caricatures inclus).

Cœur de calcul de l'extension
:mod:`~media_restorer.extensions.auto_face_id_register`, **sans dépendance
Qt** — comme :mod:`media_restorer.engines.vectorise`, ne dérive pas de
``BaseEngine`` (contrat différent : une image + une liste de libellés en
entrée, un dictionnaire de points en sortie).

Approche
--------
Un détecteur d'objets en **vocabulaire ouvert** (zero-shot) est piloté par la
liste de repères elle-même : chaque libellé devient une requête texte
(« Left Eye » → « eye »), et l'on prend le **centre de la meilleure boîte**
retournée.  Deux raisons à ce choix :

- il honore une liste de repères **configurable** (un modèle à points fixes
  imposerait un jeu figé — incompatible avec la liste partagée éditable, voir
  :mod:`media_restorer.landmark_config`) ;
- sur des **dessins/caricatures**, un détecteur généraliste en vocabulaire
  ouvert dégrade plus gracieusement qu'un réseau de landmarks entraîné
  uniquement sur des photos — et l'étape de **revue manuelle** de l'extension
  compense les imprécisions restantes.

Injection de dépendances
-------------------------
Le *détecteur* est injectable (convention du ``CLAUDE.md`` racine) :
:func:`build_detector` construit paresseusement un pipeline ``transformers``
(import lourd différé, modèle téléchargé au premier usage) ; les tests passent
un faux détecteur à :func:`detect_landmarks`, sans téléchargement ni inférence.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import cv2
import numpy as np

# Une détection = ``{"score": float, "label": str, "box": {xmin,ymin,xmax,ymax}}``
# (forme renvoyée par le pipeline ``zero-shot-object-detection`` de transformers).
Detection = dict
# Un détecteur : ``(image_bgr, requêtes) -> list[Detection]``.
Detector = Callable[[np.ndarray, "list[str]"], "list[Detection]"]

# Modèles zero-shot proposés au premier usage (l'utilisateur choisit puis le
# choix est persisté).  Du plus léger au plus précis — noms Hugging Face Hub.
CANDIDATE_MODELS: list[str] = [
    "google/owlvit-base-patch32",
    "google/owlv2-base-patch16-ensemble",
    "google/owlv2-large-patch14-ensemble",
]

_SIDE_PREFIXES = ("left", "right", "upper", "lower", "top", "bottom")


def _label_to_query(label: str) -> str:
    """Requête texte pour *label* : sans le préfixe de côté, en minuscules.

    « Left Eye » → « eye », « Right Ear » → « ear », « Nose » → « nose ».  Ainsi
    les variantes gauche/droite partagent la même requête de détection ; leur
    distinction se fait ensuite par la position (voir :func:`detect_landmarks`).
    """
    words = label.strip().lower().split()
    if words and words[0] in _SIDE_PREFIXES:
        words = words[1:]
    return " ".join(words) or label.strip().lower()


def _side_rank(label: str) -> int:
    """Rang de tri gauche→droite : ``left`` avant neutre avant ``right``."""
    words = label.lower().split()
    if "left" in words:
        return 0
    if "right" in words:
        return 2
    return 1


def _box_center(box: dict) -> tuple[int, int]:
    """Centre ``(x, y)`` (entiers) d'une boîte ``{xmin,ymin,xmax,ymax}``."""
    x = (box["xmin"] + box["xmax"]) / 2.0
    y = (box["ymin"] + box["ymax"]) / 2.0
    return int(round(x)), int(round(y))


def detect_landmarks(
    image: np.ndarray,
    labels: list[str],
    *,
    detector: Detector,
    min_score: float = 0.05,
    max_side: int | None = 1536,
) -> dict[str, tuple[int, int] | None]:
    """Localise chaque repère de *labels* dans *image* via *detector*.

    Paramètres
    ----------
    image : np.ndarray
        Image (BGR ou niveaux de gris) telle que lue par
        :func:`~media_restorer.image_io.imread_oriented`.
    labels : list[str]
        Repères à localiser (« Left Eye », « Nose »…).  L'ordre est conservé
        dans le résultat.
    detector : Detector
        ``(image, requêtes) -> détections``.  En production, issu de
        :func:`build_detector` ; en test, un faux.
    min_score : float
        Score minimal d'une détection retenue.
    max_side : int | None
        Si le plus grand côté de *image* dépasse *max_side*, la détection tourne
        sur une copie réduite à cette taille, puis les coordonnées sont
        remises à l'échelle de l'image d'origine.  Indispensable en pratique :
        un scan de plusieurs dizaines de mégapixels rendrait l'inférence
        interminable (mesuré : > 9 min CPU sur 36 Mpx), sans gain de précision
        pour localiser des repères.  ``None`` désactive la réduction.

    Retour
    ------
    dict[str, (x, y) | None]
        Pour chaque libellé, le centre pixel (dans le repère de l'image
        d'origine) de la boîte retenue, ou ``None`` si rien n'a été détecté.

    Attribution gauche/droite
    -------------------------
    Les libellés partageant une même requête (« Left Eye »/« Right Eye » → «
    eye ») se voient attribuer les meilleures boîtes de cette requête, triées
    par abscisse : la plus à gauche au libellé « left », la plus à droite au
    libellé « right ».
    """
    scaled, inv_scale = _downscale_for_detection(image, max_side)

    label_query = {label: _label_to_query(label) for label in labels}
    queries = list(dict.fromkeys(label_query.values()))

    detections = detector(scaled, queries)

    by_query: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        if float(det.get("score", 0.0)) >= min_score:
            by_query[det["label"]].append(det)

    labels_by_query: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        labels_by_query[label_query[label]].append(label)

    result: dict[str, tuple[int, int] | None] = {}
    for query, group_labels in labels_by_query.items():
        dets = sorted(by_query.get(query, []), key=lambda d: d["score"], reverse=True)
        dets = dets[: len(group_labels)]
        dets_by_x = sorted(dets, key=lambda d: _box_center(d["box"])[0])
        ordered_labels = sorted(group_labels, key=_side_rank)
        for i, label in enumerate(ordered_labels):
            if i < len(dets_by_x):
                cx, cy = _box_center(dets_by_x[i]["box"])
                result[label] = (int(round(cx * inv_scale)), int(round(cy * inv_scale)))
            else:
                result[label] = None

    # Conserve l'ordre d'origine des libellés.
    return {label: result.get(label) for label in labels}


def _downscale_for_detection(
    image: np.ndarray, max_side: int | None
) -> tuple[np.ndarray, float]:
    """Réduit *image* pour que son plus grand côté ≤ *max_side*.

    Retourne ``(image_réduite, facteur_inverse)`` où ``facteur_inverse``
    remultiplie une coordonnée de l'image réduite vers l'image d'origine
    (``1.0`` si aucune réduction).
    """
    if max_side is None:
        return image, 1.0
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    resized = cv2.resize(
        image, (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, 1.0 / scale


def build_detector(model_name: str, *, device: str = "cpu") -> Detector:
    """Construit un détecteur zero-shot réel à partir d'un pipeline ``transformers``.

    Import lourd (``transformers``, ``torch``, ``PIL``) différé ici : la
    construction — et le téléchargement du modèle au premier usage — n'a lieu
    que lorsqu'une détection réelle est demandée, jamais à l'import du module ni
    pendant les tests (qui injectent un faux détecteur).

    Paramètres
    ----------
    device : str
        ``"cpu"`` (défaut, fiable) ou ``"gpu"``.  Sur la Radeon 780M (gfx1103),
        le chargement du modèle vers le GPU ROCm peut se figer ; le CPU reste
        rapide sur l'image réduite envoyée à la détection (voir
        :mod:`media_restorer.extensions.auto_face_id_register.config`).
    """
    import os  # noqa: PLC0415

    # Radeon 780M (gfx1103) : ROCm 5.7 n'en fournit pas les noyaux, mais gfx1100
    # (même microarchitecture RDNA3) fonctionne — sans ce forçage, l'inférence
    # GPU plante (« rocBLAS error … gfx1103 »).  ``cli.py`` le pose déjà au
    # démarrage de l'application ; on le repose ici par ``setdefault`` (donc
    # sans jamais écraser un choix explicite) pour que le moteur reste correct
    # hors ``cli.py`` — script, test d'intégration, autre point d'entrée — tant
    # que torch n'a pas encore initialisé le GPU.  Voir la note détaillée dans
    # :mod:`media_restorer.cli`.
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

    from PIL import Image  # noqa: PLC0415 — import lourd volontairement différé
    from transformers import pipeline  # noqa: PLC0415

    # transformers : device=-1 → CPU, device=0 → premier GPU (cuda/ROCm).
    pipe = pipeline(
        "zero-shot-object-detection", model=model_name,
        device=0 if device == "gpu" else -1,
    )

    def detector(image: np.ndarray, queries: list[str]) -> list[Detection]:
        if image.ndim == 2:
            rgb = np.stack([image] * 3, axis=-1)
        else:
            rgb = image[..., ::-1]  # BGR (OpenCV) → RGB (PIL)
        pil = Image.fromarray(np.ascontiguousarray(rgb))
        return pipe(pil, candidate_labels=list(queries))

    return detector
