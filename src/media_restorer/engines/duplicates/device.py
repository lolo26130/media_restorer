"""Choix de l'appareil de calcul (CPU ou GPU), sondé sans risque de blocage.

Le GPU de la machine de développement — un Radeon 780M (gfx1103) — **fonctionne**
mais seulement au prix d'un réglage non officiel :

* sans ``HSA_OVERRIDE_GFX_VERSION=11.0.0``, la moindre opération échoue
  (``HIP error: invalid device function``) ;
* avec, produits matriciels et transformeurs tournent correctement, pour un gain
  mesuré de **2,2 à 2,5×** sur un ViT.

Ce gain modeste est une bonne nouvelle : le GPU est un **confort, pas un
prérequis**.  Décrire tout le corpus prend 11 minutes en CPU contre 4,6 en GPU —
un repli sur le CPU n'empêche donc rien, ce qui autorise un mode « Auto » sans
état d'âme.

.. warning::

   **La sonde s'exécute dans un SOUS-PROCESSUS, jamais dans un fil.**  C'est le
   point non négociable de ce module.  Un pilote GPU qui se bloque n'est pas
   interruptible depuis le processus qui l'a sollicité : ni ``KeyboardInterrupt``,
   ni ``thread.join(timeout)`` n'y peuvent quoi que ce soit — l'application
   entière se fige.  C'est précisément ce qui s'est produit lors du chargement
   d'OWLv2 sur ce GPU.  Un sous-processus, lui, se tue.

   Toute future sonde matérielle de ce projet doit suivre la même règle.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_GPU = "gpu"

DEVICE_TITLES = {
    DEVICE_AUTO: "Automatique (teste le GPU, retombe sur le processeur)",
    DEVICE_CPU: "Processeur seulement",
    DEVICE_GPU: "Carte graphique (forcé)",
}
DEVICE_ORDER = (DEVICE_AUTO, DEVICE_CPU, DEVICE_GPU)

#: Réglage indispensable sur gfx1103.  C'est bien ``11.0.0`` — la valeur
#: ``11.0.2``, souvent recommandée ailleurs, provoque des gels système.
HSA_OVERRIDE = "11.0.0"

#: Au-delà, on considère le GPU comme bloqué et on l'abandonne.  Large : le
#: premier appel initialise le pilote, ce qui prend plusieurs secondes.
PROBE_TIMEOUT = 45.0

#: Clé QSettings mémorisant le résultat — on ne sonde pas à chaque lancement.
SETTINGS_KEY = "duplicates/gpu_probe"

# Programme de la sonde.  Volontairement minimal : on veut savoir si le GPU
# répond, pas mesurer ses performances.
_PROBE_SOURCE = """
import json, sys
try:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "reason": "aucun GPU visible pour torch"}))
        sys.exit(0)
    nom = torch.cuda.get_device_name(0)
    a = torch.randn(256, 256, device="cuda")
    (a @ a).sum().item()          # force la synchronisation
    import torch.nn.functional as F
    x = torch.randn(1, 3, 64, 64, device="cuda")
    w = torch.randn(8, 3, 3, 3, device="cuda")
    F.conv2d(x, w).sum().item()   # les convolutions sont le point sensible
    print(json.dumps({"ok": True, "reason": nom}))
except Exception as exc:
    print(json.dumps({"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}))
"""


def probe_environment() -> dict[str, str]:
    """Environnement à passer au sous-processus (contournement inclus)."""
    env = dict(os.environ)
    env.setdefault("HSA_OVERRIDE_GFX_VERSION", HSA_OVERRIDE)
    return env


def probe_gpu(timeout: float = PROBE_TIMEOUT,
              runner=None) -> tuple[bool, str]:
    """Le GPU est-il utilisable ?  Renvoie ``(disponible, explication)``.

    Exécute un calcul témoin — produit matriciel **et** convolution — dans un
    **sous-processus** tué au bout de *timeout*.  Ne lève jamais : toute issue
    autre que le succès est un repli sur le CPU, accompagné de sa raison.

    *runner* est injectable pour les tests, qui ne doivent ni lancer de
    sous-processus ni toucher au GPU.
    """
    if runner is not None:
        return runner()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SOURCE],
            capture_output=True, text=True, timeout=timeout,
            env=probe_environment(),
        )
    except subprocess.TimeoutExpired:
        # Le cas qui justifie tout ce module : le pilote s'est bloqué.  Le
        # sous-processus est déjà tué ; l'application, elle, n'a rien senti.
        return False, f"le GPU n'a pas répondu en {timeout:.0f} s (bloqué)"
    except Exception as exc:
        return False, f"sonde impossible : {type(exc).__name__}"

    try:
        verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return False, "sonde illisible — repli sur le processeur"
    return bool(verdict.get("ok")), str(verdict.get("reason", ""))


def resolve(mode: str = DEVICE_AUTO, *,
            probe=probe_gpu,
            cached: tuple[bool, str] | None = None) -> tuple[str, str]:
    """Traduit un mode choisi par l'utilisateur en appareil torch effectif.

    Renvoie ``("cuda"|"cpu", explication)``.  L'explication est destinée à la
    barre d'état : l'utilisateur doit savoir **pourquoi** son calcul tourne où
    il tourne, surtout quand « Auto » a décidé pour lui.

    *cached* évite de resonder si le résultat est déjà connu de cette session.
    """
    if mode == DEVICE_CPU:
        return "cpu", "processeur (choix explicite)"

    if mode == DEVICE_GPU:
        # Forcé : on pose quand même le contournement, sans quoi rien ne marche.
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", HSA_OVERRIDE)
        return "cuda", "carte graphique (forcée — aucun repli en cas d'échec)"

    disponible, raison = cached if cached is not None else probe()
    if disponible:
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", HSA_OVERRIDE)
        return "cuda", f"carte graphique détectée : {raison}"
    return "cpu", f"processeur — {raison}"


def remember(available: bool, reason: str) -> None:
    """Mémorise le résultat de la sonde dans les préférences."""
    from media_restorer.app_settings import app_settings

    app_settings().setValue(SETTINGS_KEY, json.dumps([bool(available), str(reason)]))


def recall() -> tuple[bool, str] | None:
    """Résultat de sonde mémorisé, ou ``None`` s'il n'y en a pas."""
    from media_restorer.app_settings import app_settings

    brut = app_settings().value(SETTINGS_KEY)
    if not brut:
        return None
    try:
        disponible, raison = json.loads(brut)
        return bool(disponible), str(raison)
    except Exception:
        return None
