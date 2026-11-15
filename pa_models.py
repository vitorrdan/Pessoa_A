# -*- coding: utf-8 -*-
"""
PESSOA A - Contrato entre a interface e os modelos (Pessoas B e C).

A interface NAO conhece a arquitetura: ela recebe um `ModelSpec`.
Para plugar um modelo basta salvar o checkpoint no formato abaixo.

CONTRATO DO CHECKPOINT (torch.save(dict, caminho)):
  {
    "arch":        "cnn" | "resnet50" | "three_branch" | outro nome registrado,
    "state_dict":  model.state_dict(),
    "mode":        "fusion"  (3 fases)  |  "single:pre" | "single:early" | "single:late",
    "in_channels": 3 (fusao, ou ResNet50 com fase unica replicada) ou 1,
    "input_size":  256,
    "output_kind": "logit" (1 saida, sigmoid fora) | "prob" | "logit2" (2 saidas, softmax),
    "imagenet_norm": False/True  (True p/ ResNet50 pre-treinada),
    "threshold":   0.5,
    "norm":        "percentile" (padrao deste modulo),
  }
Tambem e aceito um modelo inteiro salvo com torch.save(model) (arch=None).

COMO REGISTRAR A ARQUITETURA (no arquivo de B/C, antes de carregar):
    from pa_models import register_arch
    register_arch("cnn", lambda **kw: MinhaCNN())

ALVO DO GRAD-CAM:
  - O modelo pode expor  `model.gradcam_layers`  (lista de nn.Module) -> usada direto
    (p/ 3 ramos: a ultima conv de cada ramo, ou a camada de fusao).
  - Senao, se tiver `layer4` (ResNet)  -> model.layer4
  - Senao, a ULTIMA nn.Conv2d encontrada.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from pa_config import (PHASES, INPUT_SIZE, DEFAULT_THRESHOLD, IMAGENET_MEAN,
                       IMAGENET_STD)

ARCH_BUILDERS: Dict[str, Callable] = {}


def register_arch(name: str, builder: Callable):
    """builder(**cfg) -> nn.Module (sem pesos carregados)."""
    ARCH_BUILDERS[name] = builder


@dataclass
class ModelSpec:
    name: str
    model: object                       # torch.nn.Module
    mode: str = "fusion"                # "fusion" ou "single:<fase>"
    in_channels: int = 3
    input_size: int = INPUT_SIZE
    output_kind: str = "logit"
    imagenet_norm: bool = False
    threshold: float = DEFAULT_THRESHOLD
    target_layers: List = field(default_factory=list)
    device: str = "cpu"

    @property
    def phases_used(self) -> List[str]:
        if self.mode == "fusion":
            return list(PHASES)
        return [self.mode.split(":", 1)[1]]


# --------------------------------------------------------------------------
def find_target_layers(model) -> List:
    import torch.nn as nn
    layers = getattr(model, "gradcam_layers", None)
    if layers:
        return list(layers)
    if hasattr(model, "layer4"):
        return [model.layer4]
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("Nenhuma camada convolucional encontrada para o Grad-CAM.")
    return [last]


def get_device(prefer_gpu=True) -> str:
    import torch
    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"


def load_model_spec(path: str, name: Optional[str] = None, device: Optional[str] = None
                    ) -> ModelSpec:
    """Carrega um checkpoint no formato do contrato acima."""
    import torch
    device = device or get_device()
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:                      # torch antigo sem weights_only
        ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        arch = ckpt.get("arch")
        if arch not in ARCH_BUILDERS:
            raise KeyError(f"Arquitetura '{arch}' nao registrada. Chame "
                           f"pa_models.register_arch('{arch}', builder) antes de carregar. "
                           f"Registradas: {list(ARCH_BUILDERS)}")
        cfg = {k: v for k, v in ckpt.items() if k != "state_dict"}
        model = ARCH_BUILDERS[arch](**cfg)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model, cfg = ckpt, {}              # modelo inteiro serializado
    model.to(device).eval()
    return ModelSpec(
        name=name or cfg.get("arch") or type(model).__name__,
        model=model,
        mode=cfg.get("mode", "fusion"),
        in_channels=int(cfg.get("in_channels", 3)),
        input_size=int(cfg.get("input_size", INPUT_SIZE)),
        output_kind=cfg.get("output_kind", "logit"),
        imagenet_norm=bool(cfg.get("imagenet_norm", False)),
        threshold=float(cfg.get("threshold", DEFAULT_THRESHOLD)),
        target_layers=find_target_layers(model),
        device=device,
    )


# --------------------------------------------------------------------------
# Pre-processamento: de cortes normalizados -> tensor (N, C, H, W)
# --------------------------------------------------------------------------
def build_input(spec: ModelSpec, vols: Dict[str, np.ndarray], zs: List[int]):
    """vols: fase -> (X,Y,Z) normalizado [0,1]; zs: indices de corte.
    Retorna tensor float32 (len(zs), C, S, S) no device do modelo.
    DEVE ser identico ao pre-processamento do treino (Pessoa B)."""
    import torch
    import torch.nn.functional as F
    chans = []
    for ph in spec.phases_used:
        chans.append(np.stack([vols[ph][:, :, z] for z in zs], axis=0))   # (N,X,Y)
    x = np.stack(chans, axis=1).astype(np.float32)                        # (N,C,X,Y)
    if x.shape[1] == 1 and spec.in_channels == 3:
        x = np.repeat(x, 3, axis=1)                                       # fase unica -> 3 canais
    elif x.shape[1] != spec.in_channels:
        raise ValueError(f"Modelo espera {spec.in_channels} canais, entrada tem {x.shape[1]}")
    t = torch.from_numpy(x)
    if t.shape[-1] != spec.input_size or t.shape[-2] != spec.input_size:
        t = F.interpolate(t, size=(spec.input_size, spec.input_size),
                          mode="bilinear", align_corners=False)
    if spec.imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        t = (t - mean) / std
    return t.to(spec.device)


def outputs_to_prob(out, output_kind: str):
    """Saida bruta da rede -> probabilidade de pCR, shape (N,) numpy."""
    import torch
    out = out.detach().float().reshape(out.shape[0], -1)
    if output_kind == "prob":
        p = out[:, 0]
    elif output_kind == "logit2":
        p = torch.softmax(out, dim=1)[:, 1]
    else:
        p = torch.sigmoid(out[:, 0])
    return p.cpu().numpy()


def make_predict_fn(spec: ModelSpec, vols: Dict[str, np.ndarray], batch_size: int = 16):
    """Retorna f(zs)->np.ndarray de probabilidades (uma por corte)."""
    import torch

    def predict(zs: List[int]) -> np.ndarray:
        probs = []
        with torch.no_grad():
            for i in range(0, len(zs), batch_size):
                x = build_input(spec, vols, list(zs[i:i + batch_size]))
                probs.append(outputs_to_prob(spec.model(x), spec.output_kind))
        return np.concatenate(probs) if probs else np.zeros(0)
    return predict


# --------------------------------------------------------------------------
# Modelo de demonstracao (so p/ testar a GUI antes de B e C entregarem)
# --------------------------------------------------------------------------
def make_demo_spec(mode="fusion", seed=0, device="cpu") -> ModelSpec:
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    cin = 3 if mode == "fusion" else 1

    class Demo(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(cin, 8, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(4),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(4))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 1))

        def forward(self, x):
            return self.head(self.f(x))

    m = Demo().to(device).eval()
    return ModelSpec(name=f"DEMO-{mode}", model=m, mode=mode, in_channels=cin,
                     target_layers=find_target_layers(m), device=device)
