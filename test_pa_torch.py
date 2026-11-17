
"""PESSOA A - Testes que EXIGEM torch (rodar na maquina do grupo; NAO foram
executados no ambiente onde este codigo foi escrito, que nao tinha torch).
Uso: python test_pa_torch.py"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn

from make_synthetic_data import make
from pa_gradcam import GradCAM, heat_in_mask_metrics, generate_report_figures
from pa_models import ModelSpec, find_target_layers, make_demo_spec, make_predict_fn, build_input
from pa_nii_io import DatasetIndex, load_case


class RegionNet(nn.Module):
    """Saida = media do canal 0 na regiao [8:24, 8:24]. O Grad-CAM tem de apontar p/ la."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        f = torch.relu(self.conv(x))
        return f[:, :, 8:24, 8:24].mean(dim=(1, 2, 3)).unsqueeze(1)


def main():

    net = RegionNet().eval()
    spec = ModelSpec("region", net, "fusion", 3, 64, target_layers=find_target_layers(net), device="cpu")
    assert spec.target_layers == [net.conv]


    x = torch.zeros(2, 3, 64, 64)
    x[:, :, 8:24, 8:24] = torch.rand(2, 3, 16, 16) + 0.5
    out = GradCAM(spec)(x, target_class=1)
    cams = out["cams"]
    assert cams.shape == (2, 64, 64) and cams.min() >= 0 and cams.max() <= 1 + 1e-6
    region = np.zeros((64, 64)); region[8:24, 8:24] = 1
    for c in cams:
        m = heat_in_mask_metrics(c, region)
        assert m["heat_fraction"] > 0.8, m
    assert len(net.conv._forward_hooks) == 0

    assert GradCAM(spec)(x, target_class=0)["cams"].max() == 0
    print("  OK  Grad-CAM: calor na regiao correta, hooks removidos, classe oposta")


    import torchvision
    r = torchvision.models.resnet50(weights=None)
    r.fc = nn.Linear(2048, 1)
    rs = ModelSpec("r50", r.eval(), "fusion", 3, 128, target_layers=find_target_layers(r), device="cpu")
    assert rs.target_layers == [r.layer4]
    o = GradCAM(rs)(torch.rand(1, 3, 128, 128))
    assert o["cams"].shape == (1, 128, 128) and 0 <= o["prob"][0] <= 1
    print("  OK  ResNet50: layer4 detectada, CAM 128x128")


    class TwoBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Conv2d(1, 2, 3, padding=1), nn.Conv2d(1, 2, 3, padding=1)
            self.fc = nn.Linear(4, 1)
            self.gradcam_layers = [self.a, self.b]
        def forward(self, x):
            fa, fb = torch.relu(self.a(x[:, :1])), torch.relu(self.b(x[:, 1:2]))
            return self.fc(torch.cat([fa.mean((2, 3)), fb.mean((2, 3))], 1))
    tb = TwoBranch().eval()
    ts = ModelSpec("tb", tb, "fusion", 3, 32, target_layers=find_target_layers(tb), device="cpu")
    o = GradCAM(ts)(torch.rand(2, 3, 32, 32))
    assert len(o["per_layer"]) == 2 and o["cams"].shape == (2, 32, 32)
    print("  OK  Modelo de 2 ramos: gradcam_layers usado, CAM medio")


    tmp = tempfile.mkdtemp()
    make(tmp, n=10, size=64, nz=6)
    idx = DatasetIndex(tmp)
    lc = load_case(idx.cases["P001"])
    for mode in ("fusion", "single:late"):
        sp = make_demo_spec(mode)
        sp.input_size = 64
        xin = build_input(sp, lc.vols, [0, 1, 2])
        assert xin.shape == (3, sp.in_channels, 64, 64)
        p = make_predict_fn(sp, lc.vols)(list(range(6)))
        assert p.shape == (6,) and ((p >= 0) & (p <= 1)).all()
    cases = [load_case(idx.cases[p]) for p in idx.pids()]
    out = tempfile.mkdtemp()
    sp = make_demo_spec("fusion"); sp.input_size = 64
    csv = generate_report_figures(sp, cases, out, strategy="max_area", per_group=2)
    assert os.path.exists(csv) and any(f.endswith(".png") for f in os.listdir(out))
    print("  OK  build_input/predict/relatorio com modelo demo")


    sp = make_demo_spec("single:early"); sp.input_size = 64; sp.in_channels = 3; sp.imagenet_norm = True
    xin = build_input(sp, lc.vols, [0])
    assert xin.shape == (1, 3, 64, 64)
    print("  OK  fase unica replicada em 3 canais + ImageNet norm")
    print("\nTODOS OS TESTES COM TORCH PASSARAM")


if __name__ == "__main__":
    main()
