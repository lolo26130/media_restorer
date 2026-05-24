# Media Restorer

Application de bureau PyQt6 pour la restauration de vieilles photos et films
par réseaux de neurones.  Le traitement s'effectue sur **CPU**, sur un **GPU
AMD local via ROCm**, ou sur un **GPU distant via Google Colab**.

---

## Fonctionnalités

| Moteur | Rôle | Accélération |
|---|---|---|
| **Real-ESRGAN ×4** | Agrandissement ×1 à ×8, usage général | GPU / CPU |
| **SwinIR** | Débruitage couleur (grain argentique, poussières) | GPU / CPU |
| **LaMa** | Inpainting (rayures, défauts isolés) | CPU |
| **GFPGAN v1.4** | Restauration faciale (portraits dégradés) | GPU / CPU |

- Traitement d'une image individuelle ou d'un **répertoire entier** (récursif)
- Les images déjà traitées (sous-dossiers portant le nom du moteur) sont **automatiquement exclues** du lot
- Délégation des calculs à un **GPU distant Google Colab** (T4/L4) via tunnel cloudflared
- Indicateur GPU · CPU · Colab dans la barre de statut
- Téléchargement intégré des poids des modèles

---

## Documentation

La documentation complète (guides d'installation, référence API, accélération
GPU, Google Colab) est générée par Sphinx :

```bash
# Ouvrir la documentation dans le navigateur
xdg-open docs/build/html/index.html

# Ou lancer le script fourni
./open-docs.sh
```

Pour reconstruire la documentation après modifications :

```bash
uv run sphinx-build -b html docs/source docs/build/html
```

---

## Installation

### Prérequis

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets)
- GPU AMD avec ROCm 5.7 (optionnel — CPU fonctionne sans)

### Installation standard (CPU)

```bash
git clone <repo>
cd media_restorer
uv sync
```

### Installation avec accélération GPU AMD (ROCm)

Le `pyproject.toml` pointe déjà sur l'index PyTorch ROCm 5.7 :

```bash
uv sync   # installe torch+rocm automatiquement
```

Sur les puces **gfx1103** (Radeon 780M, Ryzen 8000 série Phoenix) non
officiellement listées dans ROCm 5.7, le flag suivant est positionné
automatiquement au démarrage :

```
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### Modèles

Télécharger les poids via l'interface (menu **Modèles → Télécharger**) ou
manuellement dans `models/` :

| Fichier | Moteur | Taille |
|---|---|---|
| `RealESRGAN_x4plus.pth` | Real-ESRGAN | ~67 Mo |
| `005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth` | SwinIR | ~38 Mo |
| `GFPGANv1.4.pth` | GFPGAN | ~333 Mo |

---

## Utilisation

### Interface graphique

```bash
uv run media-restorer gui
```

### Ligne de commande

```bash
# Restaurer une photo
uv run media-restorer photo input.jpg output.jpg --engine Real-ESRGAN

# Traiter un répertoire entier
uv run media-restorer batch /chemin/dossier --recursive
```

---

## Google Colab

Pour déléguer le traitement à un GPU Google Colab (gratuit, T4 ou L4) :

1. Ouvrir `colab/server.ipynb` dans [Google Colab](https://colab.research.google.com)
2. **Runtime › Run all** — installe les librairies sur Drive (une seule fois),
   démarre le serveur FastAPI et le tunnel cloudflared
3. La dernière cellule affiche une URL du type
   `https://abc-def-123.trycloudflare.com`
4. Dans Media Restorer : **Colab › Connecter Colab** (`Ctrl+Shift+C`),
   coller l'URL
5. **Colab › Restaurer (Colab)** (`Ctrl+Shift+R`) pour traiter l'image affichée

Les modèles et librairies sont sauvegardés sur Google Drive —
seul le tunnel doit être recréé à chaque session.

---

## Architecture

```
src/media_restorer/
├── gui.py              # Fenêtre principale (PhotoRestorationGUI)
├── colab_calc.py       # Mixin Colab (ColabCalc + _ColabRestoreWorker)
├── cli.py              # Point d'entrée CLI (pose HSA_OVERRIDE_GFX_VERSION)
├── engines/
│   ├── base.py         # Interface abstraite BaseEngine
│   ├── realesrgan_engine.py
│   ├── swinir_engine.py
│   ├── lama_engine.py
│   └── gfpgan_engine.py
├── download_models.py  # Téléchargement des poids (MODEL_REGISTRY)
├── power.py            # Mode performance CPU
└── views/main.ui       # Interface Qt (compilée automatiquement)

colab/
└── server.ipynb        # Serveur FastAPI + tunnel cloudflared
```

---

## Licence

MIT — voir `LICENSE`.
