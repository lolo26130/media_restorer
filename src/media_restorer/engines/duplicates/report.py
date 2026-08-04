"""Export du graphe de doublons : CSV pour trancher, HTML pour regarder.

Deux formats, deux usages :

**CSV** — une ligne par paire, avec le dictionnaire de mérite au complet. Sert à
trier dans un tableur, à archiver une campagne, ou à comparer deux réglages.

**HTML** — les groupes rendus visuellement, avec la phrase explicative de chaque
paire.  Autonome : les vignettes sont **incorporées en base64**, si bien que le
fichier reste lisible s'il est déplacé ou envoyé, sans dossier d'images à
traîner.

Aucun des deux ne modifie les images.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Sequence

from media_restorer import tabular
from media_restorer.engines.duplicates.groups import DuplicateGraph, Pair

#: Colonnes du CSV, dans l'ordre.  L'ordre est figé : un fichier exporté
#: aujourd'hui doit rester relisable par le même script demain.
CSV_FIELDS = (
    "image_a", "image_b", "regime", "merite", "n_inliers", "ratio_inliers",
    "rotation_deg", "echelle", "anisotropie",
    "couverture_a_dans_b", "couverture_b_dans_a",
    "gain_photometrique", "offset_photometrique", "ratio_epaisseur_trait",
    "cosinus_semantique", "explication",
)

THUMB_SIDE = 220


def write_csv(path: Path | str, pairs: Sequence[Pair]) -> None:
    """Écrit une ligne par paire, dictionnaire de mérite déplié."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = tabular.dict_writer(f, CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for p in pairs:
            ligne = p.merit.to_dict()
            ligne.pop("translation_px", None)      # un couple ne va pas en CSV
            ligne["image_a"] = str(p.a)
            ligne["image_b"] = str(p.b)
            ligne["explication"] = p.merit.explain(p.a.name, p.b.name)
            writer.writerow(ligne)


def _thumb_data_uri(path: Path, side: int = THUMB_SIDE) -> str:
    """Vignette encodée en URI de données, ou chaîne vide si illisible."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.draft("RGB", (side, side))
            t = im.convert("RGB")
            t.thumbnail((side, side))
            tampon = io.BytesIO()
            t.save(tampon, format="JPEG", quality=72)
    except Exception:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(tampon.getvalue()).decode()


def write_html(path: Path | str, graph: DuplicateGraph, *, max_groups: int = 200) -> None:
    """Rapport visuel autonome : groupes, vignettes et phrases explicatives.

    *max_groups* borne la taille du fichier : une campagne sur des milliers
    d'images peut produire des centaines de groupes, et un HTML de plusieurs
    centaines de mégaoctets ne s'ouvrirait pas.
    """
    morceaux: list[str] = [_HEAD, f"<h1>Doublons — {html.escape(graph.summary())}</h1>"]

    for numero, groupe in enumerate(graph.groups[:max_groups], start=1):
        morceaux.append(f'<section><h2>Groupe {numero:03d} — {groupe.size} images</h2>')
        morceaux.append('<div class="row">')
        for membre in groupe.members:
            marque = ' <span class="rep">représentant</span>' if membre == groupe.representative else ""
            morceaux.append(
                f'<figure><img src="{_thumb_data_uri(membre)}" alt="">'
                f'<figcaption>{html.escape(membre.name)}{marque}</figcaption></figure>'
            )
        morceaux.append("</div>")
        for paire in groupe.pairs[:6]:
            morceaux.append(
                f'<p class="expl">{html.escape(paire.merit.explain(paire.a.name, paire.b.name))}</p>'
            )
        morceaux.append("</section>")

    if graph.inclusions:
        morceaux.append("<h1>Inclusions — un dessin contenu dans un autre</h1>")
        for paire in graph.inclusions[:max_groups]:
            morceaux.append('<section><div class="row">')
            for chemin in (paire.a, paire.b):
                morceaux.append(
                    f'<figure><img src="{_thumb_data_uri(chemin)}" alt="">'
                    f'<figcaption>{html.escape(chemin.name)}</figcaption></figure>'
                )
            morceaux.append("</div>")
            morceaux.append(
                f'<p class="expl">{html.escape(paire.merit.explain(paire.a.name, paire.b.name))}</p>'
            )
            morceaux.append("</section>")

    if len(graph.groups) > max_groups:
        morceaux.append(
            f"<p class='note'>… {len(graph.groups) - max_groups} groupe(s) "
            f"supplémentaire(s) non affiché(s) — voir l'export CSV.</p>"
        )
    morceaux.append("</body></html>")
    Path(path).write_text("\n".join(morceaux), encoding="utf-8")


_HEAD = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<title>Doublons — media_restorer</title>
<style>
 body{font-family:system-ui,sans-serif;margin:2rem;color:#222;background:#fafafa}
 h1{color:#1F4E79;font-size:1.4rem;margin-top:2rem}
 h2{font-size:1.05rem;margin:.4rem 0}
 section{background:#fff;border:1px solid #e3e3e3;border-radius:6px;
         padding:.8rem 1rem;margin:.8rem 0}
 .row{display:flex;flex-wrap:wrap;gap:.8rem}
 figure{margin:0;text-align:center;font-size:.8rem;max-width:230px}
 img{max-width:220px;border:1px solid #ddd;background:#fff}
 figcaption{word-break:break-all}
 .rep{color:#2E6B3E;font-weight:bold}
 .expl{font-size:.9rem;color:#444;margin:.4rem 0 0}
 .note{color:#A33A2E}
</style></head><body>"""
