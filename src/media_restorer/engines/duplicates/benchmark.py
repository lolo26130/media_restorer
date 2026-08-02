"""Banc d'essai : quel modèle reconnaît vraiment une variante redessinée ?

Je ne le sais pas a priori, et la littérature ne le dit pas : elle place DINOv2
devant CLIP en similarité fine, mais sur des **photographies**.  Sur des
caricatures redessinées, la question est ouverte.

Ce module y répond **par la mesure**, sur les verdicts que l'utilisateur a
rendus (voir :mod:`~media_restorer.engines.duplicates.verdicts`).  Chaque modèle
est évalué sur **le même jeu de paires jugées** — condition nécessaire pour que
la comparaison veuille dire quelque chose.

Le classement produit n'est pas une vérité générale sur ces modèles : c'est le
meilleur choix **pour ce corpus, sur ces verdicts**.  C'est exactement ce dont
on a besoin.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from media_restorer.engines.duplicates.verdicts import (
    MIN_VERDICTS,
    Verdict,
    calibrate,
)


def cosines_for(
    verdicts: Sequence[Verdict],
    vectors: Mapping[Path, np.ndarray],
) -> dict[str, float]:
    """Cosinus de chaque paire jugée, selon un jeu d'empreintes donné.

    Les paires dont un des deux chemins n'a pas d'empreinte sont omises : mieux
    vaut un banc d'essai portant sur moins de paires qu'un banc d'essai faussé
    par des valeurs inventées.
    """
    resultat: dict[str, float] = {}
    for v in verdicts:
        va, vb = vectors.get(Path(v.path_a)), vectors.get(Path(v.path_b))
        if va is None or vb is None:
            continue
        # Indexé par la clé QUE PORTE le verdict, non par une clé recalculée :
        # les deux doivent coïncider, mais s'y fier créerait un couplage
        # silencieux — un verdict importé ou migré n'aurait plus de cosinus.
        resultat[v.pair_key] = float(np.dot(_unit(va), _unit(vb)))
    return resultat


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def compare_models(
    verdicts: Iterable[Verdict],
    embeddings_by_model: Mapping[str, Mapping[Path, np.ndarray]],
) -> dict:
    """Compare les modèles sur le même jeu de verdicts.

    Renvoie ``{"classement": [...], "meilleur": clé|None, "message": str}``.
    Chaque entrée du classement porte la calibration complète du modèle — seuil
    suggéré, précision, rappel — de sorte que l'interface puisse afficher
    *pourquoi* un modèle l'emporte, et non seulement lequel.
    """
    juges = list(verdicts)
    if len(juges) < MIN_VERDICTS:
        return {
            "classement": [], "meilleur": None,
            "message": (f"{len(juges)} verdict(s) rendus — il en faut au moins "
                        f"{MIN_VERDICTS} pour comparer les modèles."),
        }

    classement: list[dict] = []
    for cle_modele, vecteurs in embeddings_by_model.items():
        cos = cosines_for(juges, vecteurs)
        # On rejoue les verdicts avec le cosinus de CE modèle : c'est ce qui
        # permet de comparer à jeu de paires constant.
        rejoues = [
            Verdict(pair_key=v.pair_key, confirmed=v.confirmed, model=cle_modele,
                    cosine=cos[v.pair_key], when=v.when,
                    path_a=v.path_a, path_b=v.path_b)
            for v in juges if v.pair_key in cos
        ]
        calibration = calibrate(rejoues, model=cle_modele)
        calibration["modele"] = cle_modele
        calibration["n_evalues"] = len(rejoues)
        classement.append(calibration)

    exploitables = [c for c in classement if c.get("suffisant")]
    classement.sort(key=lambda c: (c.get("suffisant", False), c.get("f1", -1.0)),
                    reverse=True)

    if not exploitables:
        return {
            "classement": classement, "meilleur": None,
            "message": ("Aucun modèle n'a pu être calibré : il faut des verdicts "
                        "des deux sortes (confirmés ET rejetés)."),
        }

    meilleur = classement[0]
    return {
        "classement": classement,
        "meilleur": meilleur["modele"],
        "message": (
            f"{meilleur['modele']} l'emporte (F1 = {meilleur['f1']:.2f}, "
            f"seuil {meilleur['seuil']:.3f}) sur {meilleur['n_evalues']} paires "
            f"jugées. Valable pour ce corpus et ces verdicts — pas une vérité "
            f"générale sur ces modèles."
        ),
    }


def format_comparison(resultat: dict) -> str:
    """Rend la comparaison en texte, pour un widget ou une console."""
    lignes = [resultat.get("message", "")]
    if resultat.get("classement"):
        lignes.append("")
        lignes.append(f"{'modèle':<16}{'F1':>7}{'seuil':>8}{'précision':>11}"
                      f"{'rappel':>9}{'paires':>8}")
        for c in resultat["classement"]:
            if not c.get("suffisant"):
                lignes.append(f"{c.get('modele', '?'):<16}   — non calibrable "
                              f"({c.get('n_evalues', 0)} paires)")
                continue
            lignes.append(
                f"{c['modele']:<16}{c['f1']:>7.2f}{c['seuil']:>8.3f}"
                f"{100 * c['precision']:>10.0f} %{100 * c['rappel']:>8.0f} %"
                f"{c['n_evalues']:>8}"
            )
    return "\n".join(lignes)
