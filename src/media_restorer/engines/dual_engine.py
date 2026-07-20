"""Moteur Double-exposition — fusion de deux prises de vue du même document.

Principe : photographier deux fois le même original sans bouger l'appareil, une
fois en éclairage **frontal** (« front light », lumière réfléchie) et une fois en
**rétroéclairage** (« back light », lumière transmise à travers le support).  Les
deux clichés portent des informations complémentaires ; ce moteur les recale puis
les recombine en une seule image de meilleure qualité.

Aucun réseau de neurones, aucun poids à télécharger : uniquement OpenCV/NumPy.
Les opérations sont vectorisées et coûtent quelques secondes sur une image de
36 Mpx — le GPU n'apporterait rien ici (voir « Coût » dans la docstring de
:class:`DualExposureEngine`).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from media_restorer.engines.base import BaseEngine
from media_restorer.image_io import imread_oriented

# Modes de fusion — l'ordre est celui proposé dans le ParameterTree.
MODE_FONDU  = "fondu"
MODE_DETAIL = "détail"
MODE_FUSION = "fusion (Mertens)"
MODE_MIN    = "min (le plus sombre)"
MODE_MAX    = "max (le plus clair)"
MODES = (MODE_DETAIL, MODE_FONDU, MODE_FUSION, MODE_MIN, MODE_MAX)

BASE_IMG1 = "image 1 (frontale)"
BASE_IMG2 = "image 2 (rétroéclairée)"

# Recalage : le décalage grossier est estimé sur une version réduite à cette
# taille max (rapide et robuste aux grands décalages, la corrélation de phase
# étant cyclique), puis affiné à pleine résolution sur un crop central.
_COARSE_MAX = 1024
_REFINE_CROP = 2048

# Fusion de Mertens : au-delà de cette taille, le calcul est réparti sur des
# tuiles chevauchantes traitées en parallèle (voir docstring de ``_mertens``).
_MERTENS_TILE = 1500
_MERTENS_TILE_PAD = 100


class DualExposureEngine(BaseEngine):
    """Moteur Double-exposition — recalage puis fusion front light / back light.

    Photographier un dessin sur papier fin (calque, papier à dessin, plaque)
    deux fois sans bouger l'appareil — une fois éclairé par l'avant, une fois
    par l'arrière — produit deux images aux défauts complémentaires :

    - **Éclairage frontal** (lumière réfléchie) : le grain du papier et les
      traits sont **nets**, mais l'image porte les reflets spéculaires, les
      salissures de surface et l'inévitable inégalité d'éclairage.
    - **Rétroéclairage** (lumière transmise) : l'éclairage est parfaitement
      uniforme et la densité mesurée est la vraie densité du crayon (le fusain
      bloque la lumière), donc le **modelé et le contraste** sont bien meilleurs ;
      mais la lumière diffuse en traversant l'épaisseur du papier, ce qui
      **atténue les hautes fréquences** : l'image est plus molle.

    D'où le mode par défaut ``détail``, qui prend la couche basse fréquence
    (tonalité, modelé, éclairage uniforme) sur le cliché rétroéclairé et y
    réinjecte la couche haute fréquence (traits, grain) du cliché frontal.
    Mesuré sur un couple 7360×4912 de cette collection : variance du laplacien
    (indice de netteté) 65 pour le rétroéclairé seul, 133 pour le frontal seul,
    87 pour la fusion — la netteté du frontal est récupérée tout en conservant
    la profondeur tonale du rétroéclairé.

    Modes de fusion (paramètre ``mode``)
    ------------------------------------
    ``détail``
        Basse fréquence de l'une + haute fréquence de l'autre (voir ci-dessus).
        Réglé par ``detail_radius``, ``detail_gain`` et ``detail_base``.
        **C'est le mode recommandé pour un couple frontal/rétroéclairé.**
    ``fondu``
        Interpolation linéaire simple pilotée par ``alpha`` (0 = image 1,
        1 = image 2).  Sert à comparer les deux prises de vue et à choisir
        visuellement un compromis ; à ``alpha=0.5`` c'est aussi une moyenne,
        qui divise le bruit de capteur par √2.
    ``fusion (Mertens)``
        Exposure fusion de Mertens et al. — combine les deux clichés en
        pondérant chaque pixel par son contraste local et son exposition.
        Générique et sans réglage fin, utile quand chaque cliché est bien
        exposé sur des zones différentes.  Attention : gourmand en mémoire
        (pyramides laplaciennes en float32 — mesuré ≈ 4 Go de pic pour un
        couple 36 Mpx).
    ``min`` / ``max``
        Minimum / maximum pixel à pixel.  ``min`` maximise la densité du tracé
        et efface tout artefact clair présent sur un seul cliché (reflet,
        poussière éclairée) ; ``max`` fait l'inverse et efface les artefacts
        sombres présents sur un seul cliché (ombre portée, poussière au dos).

    Recalage
    --------
    Même sur trépied, deux déclenchements successifs se décalent de quelques
    pixels.  ``align=True`` (défaut) estime une translation pure en deux
    passes par corrélation de phase :

    1. **Grossière** sur une version réduite (côté max ``1024``) — rapide et
       robuste aux grands décalages, la corrélation de phase étant cyclique.
    2. **Affinage** à pleine résolution sur un crop central de ``2048`` px,
       après pré-décalage de l'image 2 par l'estimation grossière.

    Mesuré sur le couple 7360×4912 : 0,29 s contre 2,15 s pour une corrélation
    directe à pleine résolution, pour un résultat identique à 0,05 px près ; un
    décalage artificiel de (+17,4 ; −9,2) px est retrouvé à 0,15 px près.

    Seule une **translation** est corrigée (c'est ce que produit un statif de
    reproduction).  Une rotation ou un changement d'échelle entre les deux
    clichés ne sera pas compensé.

    Après recalage, les deux images sont rognées de la bande de bord devenue
    invalide (quelques pixels), donc **le résultat est très légèrement plus
    petit que l'entrée**.

    Coût
    ----
    Mesuré sur un couple 7360×4912 (36 Mpx) : recalage 0,3 s ; ``fondu`` et
    ``min``/``max`` < 0,1 s ; ``détail`` ≈ 0,5 s ; ``fusion`` ≈ 1,3 s (contre
    2,9 s en un seul appel — voir « Parallélisme » ci-dessous) et ≈ 4 Go de
    mémoire vive.  Le passe-bas du mode ``détail`` est calculé sur une
    version sous-échantillonnée puis ré-agrandie — le résultat étant limité en
    bande, l'approximation est quasi exacte (écart max mesuré 2,5 niveaux sur
    255, moyenne 0,06) pour un gain de vitesse de 13×.

    Parallélisme (mode ``fusion``)
    -------------------------------
    Voir la docstring de ``_mertens`` pour le détail : au-delà de
    ``_MERTENS_TILE`` px, l'image est découpée en tuiles chevauchantes
    traitées en parallèle par des threads Python (``ThreadPoolExecutor``),
    OpenCV libérant le GIL pendant chaque appel ``MergeMertens``.  Gain
    mesuré ×2,2 sur un couple 36 Mpx, pour un écart de moins de 3 niveaux sur
    255 face au calcul plein cadre — imperceptible.

    Paramètres
    ----------
    model_path : Path | None
        Ignoré — ce moteur n'utilise aucun poids.  Accepté pour rester
        compatible avec :func:`~media_restorer.engines.build_engine`.
    second_path : str | Path
        Chemin de la **deuxième** image du couple.  La première est celle
        chargée dans la fenêtre principale.
    mode : str
        Mode de fusion, parmi :data:`MODES`.
    alpha : float
        Poids de l'image 2 dans le mode ``fondu`` (0 = image 1, 1 = image 2).
    align : bool
        Recaler l'image 2 sur l'image 1 par translation (défaut True).
    match_levels : bool
        Aligner d'abord les niveaux de l'image 2 sur ceux de l'image 1 (gain
        et offset par canal, moindres carrés).  Rend le ``fondu`` réellement
        progressif au lieu d'une simple rampe de luminosité, et neutralise la
        dominante colorée entre les deux éclairages.  Défaut False.
    detail_radius : int
        Rayon (σ gaussien, px) séparant basses et hautes fréquences en mode
        ``détail``.  Défaut 15.
    detail_gain : float
        Gain appliqué à la couche de détail.  Défaut 0.8 — au-delà de 1.0 le
        rendu durcit et les extrêmes commencent à écrêter (mesuré : 1 % de
        pixels écrêtés à gain 1.0, 3 % à gain 1.3).
    detail_base : str
        Cliché fournissant la couche basse fréquence : :data:`BASE_IMG2`
        (défaut, le rétroéclairé) ou :data:`BASE_IMG1`.
    w_contrast, w_exposure : float
        Poids de contraste et d'exposition de la fusion de Mertens.
    """

    def __init__(
        self,
        model_path:    Path | None = None,
        second_path:   str | Path  = "",
        mode:          str   = MODE_DETAIL,
        alpha:         float = 0.5,
        align:         bool  = True,
        match_levels:  bool  = False,
        detail_radius: int   = 15,
        detail_gain:   float = 0.8,
        detail_base:   str   = BASE_IMG2,
        w_contrast:    float = 1.0,
        w_exposure:    float = 0.0,
    ) -> None:
        self._second_path   = str(second_path or "")
        self._mode          = mode
        self._alpha         = float(alpha)
        self._align         = bool(align)
        self._match_levels  = bool(match_levels)
        self._detail_radius = int(detail_radius)
        self._detail_gain   = float(detail_gain)
        self._detail_base   = detail_base
        self._w_contrast    = float(w_contrast)
        self._w_exposure    = float(w_exposure)
        # Couple recalé du dernier appel — exploité par la GUI pour le fondu
        # interactif au slider sans refaire le recalage à chaque mouvement.
        self.aligned_pair: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Recalage
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_shift(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[float, float]:
        """Décalage (dx, dy) en pixels de *gray_b* par rapport à *gray_a*.

        Deux passes : corrélation de phase sur une version réduite, puis
        affinage à pleine résolution sur un crop central (voir docstring de
        classe).  Les entrées sont des images en niveaux de gris de même taille.
        """
        h, w = gray_a.shape[:2]
        scale = max(1.0, max(h, w) / _COARSE_MAX)

        # ── Passe 1 : estimation grossière sur image réduite ──
        small_a = cv2.resize(gray_a, None, fx=1 / scale, fy=1 / scale,
                             interpolation=cv2.INTER_AREA).astype(np.float32)
        small_b = cv2.resize(gray_b, None, fx=1 / scale, fy=1 / scale,
                             interpolation=cv2.INTER_AREA).astype(np.float32)
        window = cv2.createHanningWindow((small_a.shape[1], small_a.shape[0]), cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(small_a, small_b, window)
        dx, dy = dx * scale, dy * scale

        # ── Passe 2 : affinage à pleine résolution, crop central ──
        radius = min(_REFINE_CROP, h, w) // 2
        if radius < 32:
            return dx, dy
        cy, cx = h // 2, w // 2
        shifted_b = DualExposureEngine._translate(gray_b, dx, dy)
        crop_a = gray_a[cy - radius:cy + radius, cx - radius:cx + radius].astype(np.float32)
        crop_b = shifted_b[cy - radius:cy + radius, cx - radius:cx + radius].astype(np.float32)
        window = cv2.createHanningWindow((2 * radius, 2 * radius), cv2.CV_32F)
        (ex, ey), _ = cv2.phaseCorrelate(crop_a, crop_b, window)
        return dx + ex, dy + ey

    @staticmethod
    def _translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Décale *img* de (−dx, −dy) — annule un décalage mesuré (dx, dy)."""
        matrix = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        return cv2.warpAffine(
            img, matrix, (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

    def align_pair(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recale *img_b* sur *img_a* et rogne la bande de bord invalide.

        Retourne le couple recalé, tous deux de la même taille — légèrement
        plus petite que l'entrée si un décalage a été corrigé.
        """
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        dx, dy = self.estimate_shift(gray_a, gray_b)
        img_b = self._translate(img_b, dx, dy)

        # Les bords ont été comblés par réplication : on les rogne des deux
        # images pour ne conserver que la zone réellement commune.
        mx, my = math.ceil(abs(dx)), math.ceil(abs(dy))
        h, w = img_a.shape[:2]
        if (mx or my) and w > 2 * mx and h > 2 * my:
            img_a = img_a[my:h - my, mx:w - mx]
            img_b = img_b[my:h - my, mx:w - mx]
        return img_a, img_b

    # ------------------------------------------------------------------
    # Harmonisation des niveaux
    # ------------------------------------------------------------------

    @staticmethod
    def match_levels(img_ref: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Ajuste *img* sur les niveaux de *img_ref* (gain/offset par canal).

        Régression linéaire aux moindres carrés canal par canal, estimée sur un
        sous-échantillonnage 1 pixel sur 8 (largement suffisant et 64× moins
        coûteux).  Corrige à la fois la différence d'exposition et la
        dominante colorée entre les deux éclairages.
        """
        out = np.empty_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            src = img[::8, ::8, c].ravel().astype(np.float32)
            dst = img_ref[::8, ::8, c].ravel().astype(np.float32)
            if src.std() < 1e-3:                      # canal plat : rien à ajuster
                out[:, :, c] = img[:, :, c]
                continue
            gain, offset = np.polyfit(src, dst, 1)
            out[:, :, c] = img[:, :, c].astype(np.float32) * gain + offset
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Modes de fusion
    # ------------------------------------------------------------------

    @staticmethod
    def blend(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
        """Fondu linéaire : ``alpha=0`` → *img_a*, ``alpha=1`` → *img_b*."""
        return cv2.addWeighted(img_a, 1.0 - alpha, img_b, alpha, 0.0)

    @staticmethod
    def _lowpass(img: np.ndarray, radius: float) -> np.ndarray:
        """Passe-bas gaussien de rayon *radius*, calculé en sous-résolution.

        Le résultat étant limité en bande, flouter une version réduite d'un
        facteur ``radius // 4`` puis ré-agrandir est quasi exact (écart max
        mesuré 2,5 niveaux sur 255) et ~13× plus rapide qu'un
        ``GaussianBlur`` à pleine résolution.  Pour les petits rayons le
        facteur retombe à 1 et le flou exact est utilisé.
        """
        factor = max(1, int(radius // 4))
        if factor == 1:
            return cv2.GaussianBlur(img.astype(np.float32), (0, 0), radius)
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small.astype(np.float32), (0, 0), radius / factor)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def _detail_transfer(self, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """Basse fréquence d'un cliché + haute fréquence de l'autre."""
        if self._detail_base == BASE_IMG1:
            base, detail_src = img_a, img_b
        else:
            base, detail_src = img_b, img_a
        radius = max(1, self._detail_radius)
        low  = self._lowpass(base, radius)
        high = detail_src.astype(np.float32) - self._lowpass(detail_src, radius)
        return np.clip(low + self._detail_gain * high, 0, 255).astype(np.uint8)

    def _mertens_single(self, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """Un appel ``MergeMertens`` sur *img_a*/*img_b* — aucun découpage."""
        merge = cv2.createMergeMertens(self._w_contrast, 1.0, self._w_exposure)
        fused = merge.process([img_a, img_b])
        return np.clip(fused * 255.0, 0, 255).astype(np.uint8)

    def _mertens(self, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """Exposure fusion de Mertens et al. sur le couple.

        Technique d'accélération : parallélisme par tuiles via des threads
        Python, pas des processus.  ``cv2.createMergeMertens().process()``
        est un appel C++ qui, comme la quasi-totalité des fonctions OpenCV,
        *libère le GIL* pendant son exécution — plusieurs appels lancés
        depuis des threads Python distincts s'exécutent donc réellement en
        parallèle sur plusieurs cœurs, sans le coût de sérialisation d'un
        ``ProcessPoolExecutor`` (les tableaux de 36 Mpx n'ont pas à être
        recopiés entre processus).

        Pourquoi découper en tuiles alors que ``parallel_for_`` d'OpenCV
        (pthreads, confirmé actif sur cette machine) pourrait déjà utiliser
        tous les cœurs : mesuré sur un couple 4910×7358, l'appel Mertens
        direct passe de 4,87 s à seulement 2,92 s entre 1 et 16 threads
        OpenCV (×1,7) — la construction/fusion de la pyramide laplacienne de
        Mertens n'est donc pas bien parallélisée en interne.  Lancer
        plusieurs tuiles indépendantes en parallèle (chaque tuile bornant
        elle-même son nombre de threads OpenCV à 1 pour éviter la
        sursouscription) mesure ×2,2 sur la même image (2,92 s → 1,3 s).

        Le recouvrement (``_MERTENS_TILE_PAD``) donne à la pyramide de
        chaque tuile le contexte des tuiles voisines ; seule la zone utile
        de chaque tuile est recomposée dans le résultat final, comme pour
        ``LaMaEngine._restore_tiled``.  Écart mesuré face à un calcul plein
        cadre (tuile 1500 px, recouvrement 100 px) : 0,68 niveau en moyenne,
        3 niveaux maximum sur 255 — imperceptible, tout en restant nettement
        plus rapide qu'un recouvrement plus large qui n'améliore pas
        sensiblement ce résultat.

        En dessous de ``_MERTENS_TILE`` (c'est le cas de l'aperçu réduit de
        la GUI), un seul appel est fait : le découpage n'apporterait rien
        sur une image déjà petite et n'ajouterait que la latence de
        démarrage du pool de threads.
        """
        h, w = img_a.shape[:2]
        if max(h, w) <= _MERTENS_TILE:
            return self._mertens_single(img_a, img_b)

        tile, pad = _MERTENS_TILE, _MERTENS_TILE_PAD
        tiles_x, tiles_y = math.ceil(w / tile), math.ceil(h / tile)
        jobs = []
        for y in range(tiles_y):
            for x in range(tiles_x):
                in_x0, in_x1 = x * tile, min(x * tile + tile, w)
                in_y0, in_y1 = y * tile, min(y * tile + tile, h)
                pad_x0, pad_x1 = max(in_x0 - pad, 0), min(in_x1 + pad, w)
                pad_y0, pad_y1 = max(in_y0 - pad, 0), min(in_y1 + pad, h)
                jobs.append((in_x0, in_x1, in_y0, in_y1, pad_x0, pad_x1, pad_y0, pad_y1))

        def _run(job):
            in_x0, in_x1, in_y0, in_y1, px0, px1, py0, py1 = job
            crop = self._mertens_single(
                img_a[py0:py1, px0:px1], img_b[py0:py1, px0:px1]
            )
            out_x0, out_y0 = in_x0 - px0, in_y0 - py0
            interior = crop[out_y0:out_y0 + (in_y1 - in_y0), out_x0:out_x0 + (in_x1 - in_x0)]
            return in_x0, in_x1, in_y0, in_y1, interior

        output = np.empty((h, w, 3), dtype=np.uint8)
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)  # évite la sursouscription : le parallélisme vient des threads Python
        try:
            with ThreadPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 4)) as pool:
                for in_x0, in_x1, in_y0, in_y1, interior in pool.map(_run, jobs):
                    output[in_y0:in_y1, in_x0:in_x1] = interior
        finally:
            cv2.setNumThreads(previous_threads)
        return output

    def fuse(self, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """Harmonisation des niveaux puis fusion, sur un couple **déjà recalé**.

        Tout ce qui suit le recalage.  ``restore_array`` l'appelle sur le
        couple pleine résolution ; la GUI l'appelle sur une version réduite
        pour l'aperçu interactif — les deux empruntent ainsi exactement le
        même chemin de calcul, seule la taille change.
        """
        if self._match_levels:
            img_b = self.match_levels(img_a, img_b)
        return self.combine(img_a, img_b)

    def combine(self, img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        """Fusionne un couple **déjà recalé et harmonisé** selon ``mode``."""
        if self._mode == MODE_FONDU:
            return self.blend(img_a, img_b, self._alpha)
        if self._mode == MODE_DETAIL:
            return self._detail_transfer(img_a, img_b)
        if self._mode == MODE_FUSION:
            return self._mertens(img_a, img_b)
        if self._mode == MODE_MIN:
            return np.minimum(img_a, img_b)
        if self._mode == MODE_MAX:
            return np.maximum(img_a, img_b)
        raise ValueError(f"Mode de fusion inconnu : {self._mode!r} (attendu : {MODES})")

    # ------------------------------------------------------------------
    # Chargement de la seconde image
    # ------------------------------------------------------------------

    @staticmethod
    def _as_bgr8(img: np.ndarray) -> np.ndarray:
        """Normalise *img* en BGR 8 bits, 3 canaux.

        La fenêtre principale charge l'image avec ``IMREAD_UNCHANGED`` : elle
        peut donc arriver en niveaux de gris, en BGRA (PNG à canal alpha) ou
        en 16 bits par canal.  Toute l'arithmétique de fusion suppose du BGR
        8 bits sur 0–255.
        """
        if img.dtype == np.uint16:
            img = (img // 257).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _load_second(self, img_a: np.ndarray) -> np.ndarray:
        """Charge la 2ᵉ image et vérifie sa compatibilité avec *img_a*."""
        if not self._second_path:
            raise ValueError(
                "Aucune 2ᵉ image sélectionnée.\n\n"
                "Onglet « Double-exposition » → paramètre « 2ᵉ image » : "
                "choisir le second cliché du couple (l'autre éclairage)."
            )
        path = Path(self._second_path)
        if not path.exists():
            raise ValueError(f"2ᵉ image introuvable : {path}")
        # imread_oriented applique l'orientation EXIF, exactement comme le
        # chargement de l'image 1 par la fenêtre principale : sans cela, un
        # boîtier écrivant Orientation=6 livrerait ici une image transposée
        # par rapport à l'image 1.  IMREAD_COLOR garantit par ailleurs 3 canaux.
        img_b = imread_oriented(path, cv2.IMREAD_COLOR)
        if img_b is None:
            raise ValueError(f"Impossible de lire la 2ᵉ image : {path}")
        if img_b.shape[:2] != img_a.shape[:2]:
            transposed = img_b.shape[:2][::-1] == img_a.shape[:2]
            detail = (
                "\n\nLes dimensions sont exactement échangées : l'un des deux "
                "clichés est tourné de 90° par rapport à l'autre. Les remettre "
                "dans le même sens avant de les fusionner."
                if transposed else
                "\n\n(Ce moteur ne corrige qu'une translation, pas un "
                "changement d'échelle ni une rotation.)"
            )
            raise ValueError(
                "Les deux clichés doivent avoir la même taille — "
                f"image 1 : {img_a.shape[1]}×{img_a.shape[0]}, "
                f"image 2 : {img_b.shape[1]}×{img_b.shape[0]}." + detail
            )
        return img_b

    # ------------------------------------------------------------------
    # API BaseEngine
    # ------------------------------------------------------------------

    def restore_array(self, img: np.ndarray) -> np.ndarray:
        """Fusionne *img* (cliché 1) avec le cliché 2 désigné par ``second_path``."""
        img_a = self._as_bgr8(img)
        img_b = self._as_bgr8(self._load_second(img_a))

        if self._align:
            img_a, img_b = self.align_pair(img_a, img_b)

        # Mémorisé *avant* l'harmonisation des niveaux et la fusion : la GUI
        # rejoue ``fuse`` sur une réduction de ce couple à chaque changement de
        # paramètre, sans refaire le recalage (voir docstring de classe).
        self.aligned_pair = (img_a, img_b)
        return self.fuse(img_a, img_b)
