"""Tests unitaires pour les moteurs de restauration.

Ces tests n'ont pas besoin des fichiers de poids : ils testent la logique
interne (masquage LaMa, BaseEngine, factory) sans instancier les réseaux.
"""
import enum

import cv2
import numpy as np
import pytest

import media_restorer.engines.dual_engine as dual_mod
from media_restorer.engines import ENGINE_PARAMS, Engine, build_engine
from media_restorer.engines.dual_engine import DualExposureEngine
from media_restorer.engines.base import BaseEngine
from media_restorer.engines.lama_engine import LaMaEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lama(bright_thresh=245, dev_thresh=30.0, dilate_px=0) -> LaMaEngine:
    """Crée un LaMaEngine sans charger les poids (attributs directs)."""
    engine = LaMaEngine.__new__(LaMaEngine)
    engine._bright_thresh = bright_thresh
    engine._dev_thresh    = dev_thresh
    engine._dilate_px     = dilate_px
    engine._model_path    = None
    return engine


# ---------------------------------------------------------------------------
# BaseEngine
# ---------------------------------------------------------------------------

class _DoubleEngine(BaseEngine):
    """Multiplie chaque pixel par 2 (moteur factice pour les tests)."""
    def restore_array(self, img: np.ndarray) -> np.ndarray:
        return np.clip(img.astype(np.uint16) * 2, 0, 255).astype(np.uint8)


def test_base_engine_restore_file_roundtrip(tmp_path):
    """restore_file lit, transforme et écrit le résultat correctement."""
    img = np.ones((10, 10, 3), dtype=np.uint8) * 50
    src = tmp_path / "src.png"
    dst = tmp_path / "dst.png"
    cv2.imwrite(str(src), img)

    _DoubleEngine().restore_file(src, dst)

    result = cv2.imread(str(dst))
    assert result is not None
    np.testing.assert_array_equal(result, img * 2)


def test_base_engine_restore_file_creates_parent_dirs(tmp_path):
    """restore_file crée les répertoires parents si nécessaire."""
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    src = tmp_path / "img.png"
    dst = tmp_path / "sub" / "out.png"
    cv2.imwrite(str(src), img)

    _DoubleEngine().restore_file(src, dst)

    assert dst.exists()


def test_base_engine_restore_file_raises_on_missing_source(tmp_path):
    """restore_file lève ValueError si l'image source n'existe pas."""
    with pytest.raises(ValueError, match="Impossible de lire"):
        _DoubleEngine().restore_file(
            tmp_path / "nonexistent.png", tmp_path / "out.png"
        )


# ---------------------------------------------------------------------------
# LaMaEngine — masque automatique
# ---------------------------------------------------------------------------

def test_lama_mask_bright_pixel():
    """Un pixel très lumineux est inclus dans le masque (poussière)."""
    engine = _make_lama(bright_thresh=245, dilate_px=0)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[25, 25] = (250, 250, 250)

    mask = engine._auto_mask(img)

    assert mask[25, 25] == 255
    assert mask[0, 0] == 0


def test_lama_mask_dark_stripe():
    """Une rayure sombre sur fond gris est détectée comme anomalie."""
    engine = _make_lama(dev_thresh=30.0, dilate_px=0)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img[:, 50] = 60  # |60 − 128| = 68 ≥ 30

    mask = engine._auto_mask(img)

    assert mask[50, 50] == 255
    assert mask[50, 0] == 0


def test_lama_mask_dilated():
    """La dilatation étend le masque autour du pixel détecté."""
    engine_no_dil  = _make_lama(bright_thresh=245, dilate_px=0)
    engine_dil     = _make_lama(bright_thresh=245, dilate_px=3)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[25, 25] = (250, 250, 250)

    mask_no  = engine_no_dil._auto_mask(img)
    mask_dil = engine_dil._auto_mask(img)

    # Avec dilatation, plus de pixels sont masqués
    assert mask_dil.sum() >= mask_no.sum()
    # Le pixel central est masqué dans les deux cas
    assert mask_dil[25, 25] == 255


def test_lama_restore_array_returns_copy_when_no_damage():
    """Si le masque est vide, restore_array retourne une copie de l'image."""
    engine = _make_lama()
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128  # fond gris uniforme

    result = engine.restore_array(img)

    np.testing.assert_array_equal(result, img)
    assert result is not img  # copie, pas l'original


def test_lama_mask_threshold_boundary():
    """bright_thresh utilise >= : 244 non masqué, 245 masqué, sur fond uniforme.

    Le dev_thresh est fixé à 1000 pour isoler uniquement le check luminosité
    (éviter que l'anomalie médiane masque aussi le pixel à 244).
    """
    # Image uniformément à 244 → aucune brightness, aucune anomalie
    engine = _make_lama(bright_thresh=245, dev_thresh=1000.0, dilate_px=0)
    img_below = np.ones((50, 50, 3), dtype=np.uint8) * 244
    assert engine._auto_mask(img_below).max() == 0

    # Image uniformément à 245 → brightness détectée partout
    img_at = np.ones((50, 50, 3), dtype=np.uint8) * 245
    assert engine._auto_mask(img_at).max() == 255


# ---------------------------------------------------------------------------
# build_engine
# ---------------------------------------------------------------------------

def test_build_engine_raises_on_unknown_engine():
    """build_engine lève ValueError pour un type inconnu."""
    class _Fake(enum.Enum):
        UNKNOWN = "Unknown"

    with pytest.raises(ValueError, match="inconnu"):
        build_engine(_Fake.UNKNOWN)


def test_build_engine_params_passed_as_kwargs(tmp_path, monkeypatch):
    """build_engine transmet les params au constructeur du moteur."""
    captured = {}

    class _FakeEngine:
        def __init__(self, model_path=None, scale=4, tile=256):
            captured["scale"] = scale
            captured["tile"]  = tile

    import media_restorer.engines as _mod
    monkeypatch.setattr(_mod, "build_engine", lambda *a, **kw: None)

    # Test direct : appel au constructeur avec des params
    _FakeEngine(scale=2, tile=128)
    assert captured == {"scale": 2, "tile": 128}


# ---------------------------------------------------------------------------
# DualExposureEngine — recalage et fusion double-exposition
# ---------------------------------------------------------------------------

def _textured(h=300, w=300, seed=0) -> np.ndarray:
    """Image BGR texturée et reproductible (le recalage a besoin de détail)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(h, w), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 1.5)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def test_dual_estimate_shift_recovers_translation():
    """estimate_shift retrouve une translation injectée, au sous-pixel près."""
    img = cv2.cvtColor(_textured(400, 400), cv2.COLOR_BGR2GRAY)
    dx_true, dy_true = 7.0, -4.0
    matrix = np.array([[1, 0, dx_true], [0, 1, dy_true]], dtype=np.float32)
    shifted = cv2.warpAffine(img, matrix, (400, 400), borderMode=cv2.BORDER_REPLICATE)

    dx, dy = DualExposureEngine.estimate_shift(img, shifted)

    assert dx == pytest.approx(dx_true, abs=0.5)
    assert dy == pytest.approx(dy_true, abs=0.5)


def test_dual_estimate_shift_zero_on_identical_images():
    """Deux images identiques donnent un décalage nul."""
    img = cv2.cvtColor(_textured(400, 400), cv2.COLOR_BGR2GRAY)

    dx, dy = DualExposureEngine.estimate_shift(img, img)

    assert dx == pytest.approx(0.0, abs=0.2)
    assert dy == pytest.approx(0.0, abs=0.2)


def test_dual_align_pair_realigns_and_crops():
    """align_pair recale l'image 2 et rogne la bande de bord devenue invalide."""
    img_a = _textured(400, 400)
    matrix = np.array([[1, 0, 6.0], [0, 1, 3.0]], dtype=np.float32)
    img_b = cv2.warpAffine(img_a, matrix, (400, 400), borderMode=cv2.BORDER_REPLICATE)
    engine = DualExposureEngine(second_path="ignored")

    out_a, out_b = engine.align_pair(img_a, img_b)

    # Rognage : le résultat est plus petit que l'entrée
    assert out_a.shape == out_b.shape
    assert out_a.shape[0] < 400 and out_a.shape[1] < 400
    # Après recalage les deux clichés coïncident très largement
    diff_before = np.abs(img_a.astype(int) - img_b.astype(int)).mean()
    diff_after  = np.abs(out_a.astype(int) - out_b.astype(int)).mean()
    assert diff_after < diff_before / 4


def test_dual_blend_endpoints_and_midpoint():
    """blend(0) rend l'image 1, blend(1) l'image 2, blend(0.5) la moyenne."""
    img_a = np.full((10, 10, 3), 40, dtype=np.uint8)
    img_b = np.full((10, 10, 3), 200, dtype=np.uint8)

    np.testing.assert_array_equal(DualExposureEngine.blend(img_a, img_b, 0.0), img_a)
    np.testing.assert_array_equal(DualExposureEngine.blend(img_a, img_b, 1.0), img_b)
    assert DualExposureEngine.blend(img_a, img_b, 0.5)[0, 0, 0] == pytest.approx(120, abs=1)


def test_dual_combine_min_and_max():
    """Les modes min/max prennent bien le pixel le plus sombre / le plus clair."""
    img_a = np.full((8, 8, 3), 40, dtype=np.uint8)
    img_b = np.full((8, 8, 3), 200, dtype=np.uint8)
    img_a[0, 0] = 250
    img_b[0, 0] = 10

    eng_min = DualExposureEngine(second_path="x", mode=dual_mod.MODE_MIN)
    eng_max = DualExposureEngine(second_path="x", mode=dual_mod.MODE_MAX)

    assert eng_min.combine(img_a, img_b)[0, 0, 0] == 10
    assert eng_min.combine(img_a, img_b)[4, 4, 0] == 40
    assert eng_max.combine(img_a, img_b)[0, 0, 0] == 250
    assert eng_max.combine(img_a, img_b)[4, 4, 0] == 200


def test_dual_combine_rejects_unknown_mode():
    """Un mode de fusion inconnu lève une erreur explicite."""
    engine = DualExposureEngine(second_path="x", mode="inexistant")
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Mode de fusion inconnu"):
        engine.combine(img, img)


def test_dual_lowpass_matches_exact_gaussian():
    """Le passe-bas sous-échantillonné reste très proche du flou exact.

    C'est l'optimisation qui rend le mode « détail » ~13× plus rapide ; on
    vérifie que l'approximation ne dérive pas (cf. docstring de _lowpass).
    """
    img = _textured(256, 256)
    radius = 16

    approx = DualExposureEngine._lowpass(img, radius)
    exact  = cv2.GaussianBlur(img.astype(np.float32), (0, 0), radius)

    assert np.abs(approx - exact).mean() < 0.5
    assert np.abs(approx - exact).max() < 6.0


def test_dual_detail_mode_takes_low_from_base_and_high_from_other():
    """Le mode détail prend la tonalité d'un cliché et la netteté de l'autre.

    Image 1 = texture nette mais sombre ; image 2 = même texture floutée et
    claire.  Base = image 2 → le résultat doit reprendre le niveau clair de
    l'image 2 tout en retrouvant la netteté de l'image 1.
    """
    sharp_dark  = np.clip(_textured(256, 256).astype(int) - 40, 0, 255).astype(np.uint8)
    blurry_pale = cv2.GaussianBlur(
        np.clip(sharp_dark.astype(int) + 80, 0, 255).astype(np.uint8), (0, 0), 3
    )
    engine = DualExposureEngine(
        second_path="x", mode=dual_mod.MODE_DETAIL,
        detail_base=dual_mod.BASE_IMG2, detail_radius=15, detail_gain=1.0,
    )

    result = engine.combine(sharp_dark, blurry_pale)

    def sharpness(i):
        return cv2.Laplacian(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    # Tonalité : proche de l'image 2 (claire), pas de l'image 1 (sombre)
    assert abs(result.mean() - blurry_pale.mean()) < abs(result.mean() - sharp_dark.mean())
    # Netteté : nettement au-dessus de l'image 2 floue dont vient la base
    assert sharpness(result) > sharpness(blurry_pale) * 2


def test_dual_match_levels_aligns_exposure():
    """match_levels ramène les niveaux de l'image 2 sur ceux de l'image 1."""
    img_a = _textured(128, 128)
    img_b = np.clip(img_a.astype(float) * 0.6 + 30, 0, 255).astype(np.uint8)

    matched = DualExposureEngine.match_levels(img_a, img_b)

    assert abs(int(matched.mean()) - int(img_a.mean())) < abs(int(img_b.mean()) - int(img_a.mean()))
    assert abs(int(matched.mean()) - int(img_a.mean())) <= 2


def test_dual_restore_array_without_second_path_explains_how_to_fix():
    """Sans 2ᵉ image, l'erreur indique où la sélectionner."""
    engine = DualExposureEngine()

    with pytest.raises(ValueError, match="2ᵉ image"):
        engine.restore_array(np.zeros((10, 10, 3), dtype=np.uint8))


def test_dual_restore_array_rejects_missing_file(tmp_path):
    """Un chemin de 2ᵉ image inexistant est signalé clairement."""
    engine = DualExposureEngine(second_path=str(tmp_path / "absent.png"))

    with pytest.raises(ValueError, match="introuvable"):
        engine.restore_array(np.zeros((10, 10, 3), dtype=np.uint8))


def test_dual_restore_array_rejects_size_mismatch(tmp_path):
    """Deux clichés de tailles différentes sont refusés (pas de mise à l'échelle)."""
    second = tmp_path / "b.png"
    cv2.imwrite(str(second), _textured(64, 64))
    engine = DualExposureEngine(second_path=str(second))

    with pytest.raises(ValueError, match="même taille"):
        engine.restore_array(_textured(128, 128))


def test_dual_restore_array_end_to_end_exposes_aligned_pair(tmp_path):
    """restore_array produit une image et mémorise le couple recalé.

    Le couple mémorisé est ce qui permet à la GUI de refaire un fondu au
    slider sans relancer le recalage.
    """
    img_a  = _textured(256, 256)
    second = tmp_path / "b.png"
    cv2.imwrite(str(second), cv2.GaussianBlur(img_a, (0, 0), 2))
    engine = DualExposureEngine(second_path=str(second), mode=dual_mod.MODE_FONDU)

    result = engine.restore_array(img_a)

    assert result.dtype == np.uint8 and result.ndim == 3
    assert engine.aligned_pair is not None
    pair_a, pair_b = engine.aligned_pair
    assert pair_a.shape == pair_b.shape == result.shape


def test_dual_engine_params_match_constructor_signature():
    """Chaque paramètre du 5ᵉ onglet correspond à un kwarg du constructeur.

    build_engine transmet les params du ParameterTree en **kwargs : une clé
    orpheline ferait planter la construction du moteur au clic sur Restaurer.
    """
    import inspect
    accepted = set(inspect.signature(DualExposureEngine.__init__).parameters) - {"self"}
    declared = {p["name"] for p in ENGINE_PARAMS[Engine.DUAL]}

    assert declared <= accepted, f"paramètres orphelins : {declared - accepted}"


def test_build_engine_dual_forwards_params(tmp_path):
    """build_engine(Engine.DUAL) construit bien un DualExposureEngine réglé."""
    engine = build_engine(
        Engine.DUAL, None, {"mode": dual_mod.MODE_MIN, "second_path": "x", "alpha": 0.25}
    )

    assert isinstance(engine, DualExposureEngine)
    assert engine._mode == dual_mod.MODE_MIN
    assert engine._alpha == 0.25


def _write_jpeg_with_exif_orientation(path, img, orientation=6):
    """Écrit *img* en JPEG en y posant un tag EXIF Orientation.

    Reproduit ce que produit un boîtier tenu verticalement : les pixels sont
    stockés en paysage et l'EXIF demande une rotation de 90°.
    """
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    exif = pil.getexif()
    exif[274] = orientation                     # 274 = Orientation
    pil.save(str(path), "JPEG", exif=exif, quality=95)


def test_dual_second_image_ignores_exif_orientation_like_the_gui(tmp_path):
    """La 2ᵉ image est lue avec la même convention EXIF que la fenêtre principale.

    Régression : la GUI et le moteur lisaient la même photo avec des drapeaux
    OpenCV traitant l'orientation EXIF différemment.  Sur un fichier
    Orientation=6 les deux clichés arrivaient transposés l'un par rapport à
    l'autre et la fusion échouait.  Les deux passent désormais par
    :func:`~media_restorer.image_io.imread_oriented`.
    """
    img = _textured(120, 200)                   # non carrée : la transposition se voit
    first  = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    _write_jpeg_with_exif_orientation(first, img)
    _write_jpeg_with_exif_orientation(second, img)

    # Exactement ce que fait PhotoRestorationGUI._load_image
    from media_restorer.image_io import imread_oriented
    img_a = imread_oriented(first)
    assert img_a.shape[:2] == (200, 120)         # redressé : 120×200 sur le disque
    # align=False pour isoler la question de l'orientation du rognage de recalage
    engine = DualExposureEngine(
        second_path=str(second), mode=dual_mod.MODE_FONDU, align=False
    )

    result = engine.restore_array(img_a)         # ne doit pas lever

    assert result.shape[:2] == img_a.shape[:2]
    # Les deux clichés étant issus des mêmes pixels, le fondu doit les
    # retrouver identiques — ce qui ne serait pas le cas si l'un était tourné.
    assert np.abs(result.astype(int) - img_a.astype(int)).mean() < 2.0


def test_dual_size_mismatch_names_the_rotation_case(tmp_path):
    """Des dimensions échangées sont signalées comme une rotation de 90°."""
    second = tmp_path / "b.png"
    cv2.imwrite(str(second), _textured(200, 120))
    engine = DualExposureEngine(second_path=str(second))

    with pytest.raises(ValueError, match="tourné de 90"):
        engine.restore_array(_textured(120, 200))


@pytest.mark.parametrize("make, label", [
    (lambda: cv2.cvtColor(_textured(64, 64), cv2.COLOR_BGR2GRAY), "niveaux de gris"),
    (lambda: cv2.cvtColor(_textured(64, 64), cv2.COLOR_BGR2BGRA), "BGRA"),
    (lambda: (_textured(64, 64).astype(np.uint16) * 257), "16 bits"),
])
def test_dual_accepts_unusual_pixel_formats(tmp_path, make, label):
    """IMREAD_UNCHANGED peut livrer du gris, du BGRA ou du 16 bits : tous acceptés."""
    second = tmp_path / "b.png"
    cv2.imwrite(str(second), _textured(64, 64))
    engine = DualExposureEngine(second_path=str(second), mode=dual_mod.MODE_FONDU)

    result = engine.restore_array(make())

    assert result.dtype == np.uint8, label
    assert result.ndim == 3 and result.shape[2] == 3, label


# ---------------------------------------------------------------------------
# DualExposureEngine — Mertens tuilé/parallèle (accélération)
# ---------------------------------------------------------------------------

def test_dual_mertens_below_tile_threshold_uses_a_single_call():
    """Sous le seuil de tuilage, aucun découpage : un seul appel Mertens."""
    from unittest.mock import patch

    engine = DualExposureEngine(second_path="x", mode=dual_mod.MODE_FUSION)
    img_a, img_b = _textured(300, 300), _textured(300, 300, seed=1)

    with patch.object(
        engine, "_mertens_single", wraps=engine._mertens_single
    ) as spy:
        engine.combine(img_a, img_b)

    assert spy.call_count == 1


def test_dual_mertens_above_tile_threshold_dispatches_several_tiles():
    """Au-delà du seuil, le calcul est réparti sur plusieurs tuiles."""
    from unittest.mock import patch

    engine = DualExposureEngine(second_path="x", mode=dual_mod.MODE_FUSION)
    size = dual_mod._MERTENS_TILE + 200
    img_a, img_b = _textured(size, size), _textured(size, size, seed=1)

    with patch.object(
        engine, "_mertens_single", wraps=engine._mertens_single
    ) as spy:
        engine.combine(img_a, img_b)

    assert spy.call_count > 1


def test_dual_mertens_tiled_matches_single_call_closely():
    """Le résultat tuilé/parallèle reste très proche d'un calcul plein cadre.

    Le découpage introduit un recouvrement pour donner du contexte à la
    pyramide laplacienne de chaque tuile ; seule la zone utile de chaque
    tuile est recomposée.  Un écart de quelques niveaux est attendu à la
    jointure des tuiles, mais il doit rester imperceptible.
    """
    engine = DualExposureEngine(second_path="x", mode=dual_mod.MODE_FUSION)
    size = dual_mod._MERTENS_TILE + 400
    img_a = _textured(size, size)
    img_b = np.clip(img_a.astype(int) + 40, 0, 255).astype(np.uint8)

    tiled = engine._mertens(img_a, img_b)
    reference = engine._mertens_single(img_a, img_b)

    diff = np.abs(tiled.astype(int) - reference.astype(int))
    assert diff.mean() < 2.0
    assert diff.max() <= 8


def test_dual_mertens_tiled_covers_the_full_frame_without_gaps():
    """Chaque pixel de sortie provient d'exactement une tuile — pas de trous."""
    engine = DualExposureEngine(second_path="x", mode=dual_mod.MODE_FUSION)
    size = dual_mod._MERTENS_TILE + 250
    img_a, img_b = _textured(size, size), _textured(size, size, seed=2)

    result = engine._mertens(img_a, img_b)

    assert result.shape == (size, size, 3)
    assert result.dtype == np.uint8


def test_dual_mertens_restores_global_opencv_thread_count():
    """Le tuilage ne doit pas laisser cv2.setNumThreads modifié après coup.

    ``cv2.setNumThreads`` est un état global du processus : l'abaisser à 1
    pour éviter la sursouscription pendant le tuilage puis oublier de le
    restaurer affecterait silencieusement tous les traitements suivants
    (y compris ceux d'autres moteurs).
    """
    import cv2

    engine = DualExposureEngine(second_path="x", mode=dual_mod.MODE_FUSION)
    size = dual_mod._MERTENS_TILE + 200
    img_a, img_b = _textured(size, size), _textured(size, size, seed=3)
    cv2.setNumThreads(7)

    try:
        engine._mertens(img_a, img_b)
        assert cv2.getNumThreads() == 7
    finally:
        cv2.setNumThreads(-1)  # revient à l'auto-détection par défaut
