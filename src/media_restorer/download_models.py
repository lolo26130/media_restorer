"""Téléchargement des poids des modèles de restauration.

Tous les modèles sont déposés dans leurs répertoires exacts attendus par
les moteurs, relatifs à la racine du projet.

Usage CLI ::

    python -m media_restorer.download_models
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).parents[2]  # src/media_restorer/../../ → racine projet

# (nom_affiché, chemin_absolu_destination, url_directe)
MODEL_REGISTRY: list[tuple[str, Path, str]] = [
    (
        "RealESRGAN x4plus",
        _ROOT / "models" / "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    (
        "SwinIR colorDN noise25",
        _ROOT / "models" / "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
    ),
    (
        "GFPGANv1.4",
        _ROOT / "models" / "GFPGANv1.4.pth",
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    ),
    (
        "LaMa big-lama",
        _ROOT / "models" / "big-lama.pt",
        "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
    ),
    (
        "GFPGAN detection (RetinaFace)",
        _ROOT / "gfpgan" / "weights" / "detection_Resnet50_Final.pth",
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    ),
    (
        "GFPGAN parsing (ParseNet)",
        _ROOT / "gfpgan" / "weights" / "parsing_parsenet.pth",
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
    ),
]

# Callback : (nom, block_num, block_size, total_octets, index_registry)
ProgressCB = Callable[[str, int, int, int, int], None]


def download_all(
    progress_cb: ProgressCB | None = None,
    skip_existing: bool = True,
) -> list[str]:
    """Télécharge tous les modèles manquants.

    Les fichiers sont écrits dans un ``.tmp`` temporaire puis renommés
    en destination finale pour éviter les fichiers corrompus en cas
    d'interruption.

    Parameters
    ----------
    progress_cb:
        Appelé à chaque bloc reçu :
        ``(nom, block_num, block_size, total, index)``.
    skip_existing:
        Si True (défaut), ignore les fichiers déjà présents.

    Returns
    -------
    list[str]
        Noms des modèles effectivement téléchargés.
    """
    downloaded: list[str] = []
    for i, (name, dest, url) in enumerate(MODEL_REGISTRY):
        if skip_existing and dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            def _hook(block_num: int, block_size: int, total: int) -> None:
                if progress_cb is not None:
                    progress_cb(name, block_num, block_size, total, i)
            urllib.request.urlretrieve(url, tmp, reporthook=_hook)
            tmp.rename(dest)
            downloaded.append(name)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    return downloaded


if __name__ == "__main__":
    print(f"Racine projet : {_ROOT}\n")
    print("État des modèles :")
    for name, dest, url in MODEL_REGISTRY:
        status = "présent  " if dest.exists() else "MANQUANT "
        print(f"  {status}  {name:40s}  {dest.relative_to(_ROOT)}")

    missing = [(n, d, u) for n, d, u in MODEL_REGISTRY if not d.exists()]
    if not missing:
        print("\nTous les modèles sont déjà présents.")
    else:
        print(f"\n{len(missing)} modèle(s) à télécharger…")
        for name, dest, url in missing:
            print(f"\n  ↓ {name}\n    {url}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            try:
                def _cli_hook(block_num: int, block_size: int, total: int) -> None:
                    if total > 0:
                        done = min(block_num * block_size, total)
                        pct  = done * 100 // total
                        bar  = "#" * (pct // 5) + "-" * (20 - pct // 5)
                        print(f"\r    [{bar}] {pct:3d}%  {done/1_048_576:.1f} Mo",
                              end="", flush=True)
                urllib.request.urlretrieve(url, tmp, reporthook=_cli_hook)
                tmp.rename(dest)
                print(f"\r    ✓ → {dest.relative_to(_ROOT)}" + " " * 20)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                print(f"\r    ✗ Erreur : {exc}")
        print("\nFin.")
