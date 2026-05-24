"""Command-line interface for media-restorer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Stratégie générale — activation GPU AMD (ROCm)
# ------------------------------------------------
# PyTorch ROCm choisit le backend GPU via l'API HSA.  Certaines puces AMD
# récentes ne figurent pas encore dans la liste de support officielle de
# ROCm mais sont binairemement compatibles avec une version antérieure.
# HSA_OVERRIDE_GFX_VERSION substitue l'identifiant de la puce au moment
# de la sélection des noyaux de calcul.  La variable est positionnée via
# setdefault : une valeur déjà présente dans l'environnement est respectée.
#
# Spécifique à cette machine — AMD Radeon 780M (Ryzen 8845HS)
# ------------------------------------------------------------
# La 780M est identifiée comme gfx1103 (RDNA3 iGPU, Phoenix).  ROCm 5.7
# ne liste officiellement que jusqu'à gfx1102.  En simulant gfx1100 les
# noyaux compilés pour RX 7900 (gfx1100) s'exécutent sans erreur sur
# gfx1103, les deux partageant la même microarchitecture RDNA3.
# Gain mesuré sur RealESRGAN (photo 602×596, 9 tuiles 256 px) : ×10 vs CPU.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="media-restorer",
        description="Restore old photos and films using Real-ESRGAN (CPU).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- photo sub-command ---
    photo = sub.add_parser("photo", help="Restore a single photo.")
    photo.add_argument("input", type=Path, help="Input image path.")
    photo.add_argument("output", type=Path, help="Output image path.")
    photo.add_argument("--model", type=Path, default=None, help="Path to .pth weights.")
    photo.add_argument("--scale", type=int, default=4, help="Upscale factor (default: 4).")

    # --- gui sub-command ---
    gui = sub.add_parser("gui", help="Launch the graphical interface.")
    gui.add_argument("--model", type=Path, default=None, help="Path to .pth weights.")

    # --- film sub-command (future) ---
    film = sub.add_parser("film", help="Restore a video file (not yet implemented).")
    film.add_argument("input", type=Path, help="Input video path.")
    film.add_argument("output", type=Path, help="Output video path.")
    film.add_argument("--model", type=Path, default=None, help="Path to .pth weights.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point registered in pyproject.toml."""
    args = _parse_args(argv)

    if args.command == "photo":
        from media_restorer.restore import restore_image

        out = restore_image(args.input, args.output, args.model, args.scale)
        print(f"Saved: {out}")

    elif args.command == "gui":
        from media_restorer.gui import run_gui

        run_gui(model_path=args.model)

    elif args.command == "film":
        from media_restorer.film import restore_film

        try:
            out = restore_film(args.input, args.output, args.model)
            print(f"Saved: {out}")
        except NotImplementedError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
