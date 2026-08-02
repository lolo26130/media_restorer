"""Verdicts de la revue — la vérité terrain que le corpus ne fournit pas.

Les régimes R1 et R2 se valident tout seuls : une homographie existe, ou elle
n'existe pas.  **R3 n'a aucune vérité terrain** — rien, dans un fonds de
dessins, ne dit quelles paires sont des variantes redessinées l'une de l'autre.
Et le seuil de similarité ne peut pas être fixé au jugé : la v1 l'a montré, sur
des dessins au trait les descripteurs globaux se ressemblent tous (32 « paires
incertaines » sur 36 candidates).

D'où ce module : chaque verdict rendu à la revue est enregistré, et l'ensemble
sert à **régler le seuil sur des données** plutôt que sur une intuition.  Le jeu
ainsi constitué resservira à tout modèle essayé plus tard — c'est le même
investissement qui sert plusieurs fois.

Clé de paire
------------
Deux chemins absolus, triés puis hachés.  Triés pour que le verdict soit
**symétrique** (juger « A, B » vaut pour « B, A ») ; hachés pour que la clé
reste courte et utilisable comme nom de champ JSON.  Un verdict doit survivre à
un reclassement du corpus : c'est le contenu du chemin qui compte, pas l'ordre
dans lequel la campagne a présenté la paire.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

_FILENAME = "duplicates_verdicts.json"
_FORMAT_VERSION = 1

#: En deçà, un seuil calibré n'a aucune valeur statistique.  L'interface s'en
#: sert pour n'activer le banc d'essai qu'à partir du moment où il dit
#: réellement quelque chose.
MIN_VERDICTS = 10


@dataclass(frozen=True)
class Verdict:
    """Le jugement humain sur une paire, et le contexte qui l'a produit."""

    pair_key: str
    confirmed: bool
    model: str
    cosine: float
    when: str
    path_a: str = ""
    path_b: str = ""


def pair_key(a: Path | str, b: Path | str) -> str:
    """Clé stable et symétrique d'une paire de chemins."""
    couple = sorted((str(Path(a).resolve()), str(Path(b).resolve())))
    return hashlib.sha1("\n".join(couple).encode("utf-8")).hexdigest()[:16]


def store_path() -> Path:
    """Emplacement du fichier de verdicts (à côté du ``.ini`` du ``QSettings``)."""
    from media_restorer.app_settings import app_settings

    return Path(app_settings().fileName()).with_name(_FILENAME)


def load(path: Path | None = None) -> dict[str, Verdict]:
    """Verdicts enregistrés, indexés par clé de paire.

    Ne lève jamais : un fichier absent ou abîmé revient à « aucun verdict ».
    Perdre des verdicts est ennuyeux, mais planter au lancement le serait plus.
    """
    path = path or store_path()
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(blob, dict) or blob.get("version") != _FORMAT_VERSION:
        return {}
    resultat: dict[str, Verdict] = {}
    for cle, brut in (blob.get("verdicts") or {}).items():
        try:
            resultat[cle] = Verdict(
                pair_key=str(cle),
                confirmed=bool(brut["confirmed"]),
                model=str(brut.get("model", "")),
                cosine=float(brut.get("cosine", 0.0)),
                when=str(brut.get("when", "")),
                path_a=str(brut.get("path_a", "")),
                path_b=str(brut.get("path_b", "")),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return resultat


def save(verdicts: dict[str, Verdict], path: Path | None = None) -> None:
    """Écrit l'ensemble des verdicts."""
    path = path or store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": _FORMAT_VERSION,
             "verdicts": {k: asdict(v) for k, v in verdicts.items()}},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )


def record(a: Path, b: Path, confirmed: bool, *, model: str, cosine: float,
           path: Path | None = None) -> Verdict:
    """Enregistre un verdict et le renvoie.

    Un nouveau jugement sur la même paire **remplace** le précédent : l'humain a
    le droit de se raviser, et c'est son dernier avis qui fait foi.
    """
    tous = load(path)
    cle = pair_key(a, b)
    verdict = Verdict(
        pair_key=cle, confirmed=bool(confirmed), model=str(model),
        cosine=float(cosine),
        when=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        path_a=str(a), path_b=str(b),
    )
    tous[cle] = verdict
    save(tous, path)
    return verdict


def verdict_for(a: Path, b: Path, path: Path | None = None) -> Verdict | None:
    """Verdict déjà rendu sur cette paire, ou ``None``."""
    return load(path).get(pair_key(a, b))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(verdicts: Iterable[Verdict], *, model: str | None = None) -> dict:
    """Cherche le seuil de cosinus qui sépare le mieux confirmés et rejetés.

    Renvoie un dictionnaire lisible tel quel par l'interface.  Le critère
    optimisé est le **score F1** : sur cette tâche, rater une variante et en
    inventer une sont deux erreurs également gênantes — la première fait perdre
    l'information, la seconde fait perdre du temps de revue.

    Renvoie ``{"suffisant": False, ...}`` tant qu'il y a trop peu de verdicts :
    annoncer un seuil calibré sur cinq exemples serait trompeur.
    """
    retenus = [v for v in verdicts if model is None or v.model == model]
    positifs = [v.cosine for v in retenus if v.confirmed]
    negatifs = [v.cosine for v in retenus if not v.confirmed]

    if len(retenus) < MIN_VERDICTS or not positifs or not negatifs:
        return {
            "suffisant": False, "n_verdicts": len(retenus),
            "n_confirmes": len(positifs), "n_rejetes": len(negatifs),
            "minimum_requis": MIN_VERDICTS,
            "message": (f"{len(retenus)} verdict(s) — il en faut au moins "
                        f"{MIN_VERDICTS}, dont des deux sortes, pour régler un seuil."),
        }

    meilleur = {"f1": -1.0}
    for candidat in sorted({round(v.cosine, 3) for v in retenus}):
        vrais_positifs = sum(1 for c in positifs if c >= candidat)
        faux_positifs = sum(1 for c in negatifs if c >= candidat)
        faux_negatifs = len(positifs) - vrais_positifs
        precision = vrais_positifs / (vrais_positifs + faux_positifs) if (vrais_positifs + faux_positifs) else 0.0
        rappel = vrais_positifs / len(positifs) if positifs else 0.0
        f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) else 0.0
        if f1 > meilleur["f1"]:
            meilleur = {"f1": f1, "seuil": candidat, "precision": precision,
                        "rappel": rappel, "vrais_positifs": vrais_positifs,
                        "faux_positifs": faux_positifs, "faux_negatifs": faux_negatifs}

    meilleur.update({
        "suffisant": True, "n_verdicts": len(retenus),
        "n_confirmes": len(positifs), "n_rejetes": len(negatifs),
        "model": model or "tous",
    })
    return meilleur


def describe(calibration: dict) -> str:
    """Rend une calibration en une phrase lisible."""
    if not calibration.get("suffisant"):
        return calibration.get("message", "Calibration impossible.")
    return (
        f"Seuil suggéré {calibration['seuil']:.3f} — "
        f"précision {100 * calibration['precision']:.0f} %, "
        f"rappel {100 * calibration['rappel']:.0f} % "
        f"(sur {calibration['n_verdicts']} verdicts : "
        f"{calibration['n_confirmes']} confirmés, {calibration['n_rejetes']} rejetés)."
    )
