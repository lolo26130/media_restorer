"""Compatibility shims importés avant basicsr dans chaque moteur.

Patches appliqués :

- torchvision >= 0.17 a supprimé ``functional_tensor`` ; on le recrée.
- torch.meshgrid sans ``indexing=`` produit un UserWarning dans torch >= 1.10 ;
  le comportement legacy (indexing='ij') est conservé par BasicSR — on filtre
  l'avertissement car le correctif est dans BasicSR, pas dans notre code.
"""

import sys
import types
import warnings

# BasicSR appelle torch.meshgrid sans indexing= (swinir_arch.py, arch_util.py).
# Le comportement implicite 'ij' est celui voulu ; on filtre l'avertissement
# de dépréciation jusqu'à ce que BasicSR soit corrigé.
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release",
    category=UserWarning,
)

import torchvision.transforms.functional as _F

if "torchvision.transforms.functional_tensor" not in sys.modules:
    _ft = types.ModuleType("torchvision.transforms.functional_tensor")
    _ft.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _ft
