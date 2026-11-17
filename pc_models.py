"""
PESSOA C - Modelos (item d do enunciado).

[Nome, matricula, curso e campus dos componentes do grupo: preencher na integracao]

Tres redes, todas devolvendo UM logit por imagem (probabilidade = sigmoid(logit)):

  1. PCCNN                   CNN propria, 6 conv + 2 FC = 8 camadas COM parametros
  2. PCResNet50              ResNet50 (ImageNet) + cabeca densa nova (1 saida)
  3. PCMultiBranchResNet18   ResNet18 com 1 ramo POR FASE e fusao no final
                             (3 ramos na fusao; 1 ramo quando e fase unica)

Contrato com a Pessoa A (pa_models.py):
  * cada arquitetura e registrada com pa_models.register_arch(nome, builder);
  * o builder recebe **todas** as chaves do checkpoint (arch, mode, in_channels, ...)
    e ignora as que nao usa (**kw);
  * a NORMALIZACAO ImageNet fica DENTRO dos modelos pre-treinados (buffers), entao o
    checkpoint grava imagenet_norm=False e a GUI so precisa entregar [0,1]
    (normalizacao por percentil 1-99 por volume/fase, igual ao pa_nii_io);
  * Grad-CAM: ResNet50 expoe .layer4; CNN expoe .gradcam_layers=[ultimo bloco conv];
    modelo de ramos expoe .gradcam_layers=[layer4 de cada ramo].
"""
import torch
import torch.nn as nn

from pa_config import IMAGENET_MEAN, IMAGENET_STD
from pa_models import register_arch

GRAY_MEAN, GRAY_STD = 0.449, 0.226


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))


class _Normalize(nn.Module):
    """(x - mean) / std com buffers (acompanham o state_dict e o device)."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def _make_head(in_features, hidden, dropout):
    """Cabeca densa: Dropout -> Linear -> ReLU -> Dropout -> Linear(1)."""
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, hidden),
                         nn.ReLU(inplace=False), nn.Dropout(dropout), nn.Linear(hidden, 1))


class _FreezeMixin:
    """Congelar/descongelar o backbone (transfer learning em 2 etapas)."""

    def _backbone_parts(self):
        raise NotImplementedError

    def _last_block_parts(self):
        raise NotImplementedError

    def freeze_backbone(self):
        for m in self._backbone_parts():
            for p in m.parameters():
                p.requires_grad = False
        self.train(self.training)

    def unfreeze_last_block(self):
        for m in self._last_block_parts():
            for p in m.parameters():
                p.requires_grad = True
        self.train(self.training)

    def train(self, mode=True):
        """BatchNorm de partes CONGELADAS fica em eval (mantem as estatisticas do
        ImageNet; com lotes pequenos, atualizar a media/variancia prejudicaria)."""
        super().train(mode)
        if mode:
            for m in self._backbone_parts():
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self


CNN_SPEC = [(16, 5, True), (32, 3, True), (64, 3, True),
            (128, 3, True), (128, 3, True), (256, 3, False)]


class PCCNN(nn.Module):
    """Entrada (N, C, 256, 256), C=1 (fase unica) ou 3 (fusao por canais).

    Conv(5x5,16)-BN-ReLU-Pool | Conv(3x3,32)-BN-ReLU-Pool | Conv(3x3,64)-BN-ReLU-Pool |
    Conv(3x3,128)-BN-ReLU-Pool | Conv(3x3,128)-BN-ReLU-Pool | Conv(3x3,256)-BN-ReLU |
    GlobalAvgPool | Dropout | FC(256->hidden)-ReLU | Dropout | FC(hidden->1)
    """

    def __init__(self, in_channels=1, dropout=0.5, hidden=64, spec=CNN_SPEC):
        super().__init__()
        blocks, c = [], in_channels
        for out_c, k, pool in spec:
            layers = [nn.Conv2d(c, out_c, k, padding=k // 2, bias=False),
                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=False)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            blocks.append(nn.Sequential(*layers))
            c = out_c
        self.blocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = _make_head(c, hidden, dropout)
        self.gradcam_layers = [self.blocks[-1]]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(self.gap(x).flatten(1))


def conv_out(w, k, p=0, s=1):
    """Formula dos slides: W_out = floor((W - K + 2P) / S) + 1."""
    return (w - k + 2 * p) // s + 1


def cnn_layer_table(in_channels=1, input_size=256, hidden=64, spec=CNN_SPEC):
    """Tabela calculada A MAO (formula acima). Retorna (linhas, shapes_apos_cada_bloco).
    Linhas: (camada, descricao, saida (C,H,W), n_parametros)."""
    rows, block_shapes = [], []
    w, c = input_size, in_channels
    rows.append(("Entrada", "", (c, w, w), 0))
    for i, (o, k, pool) in enumerate(spec, 1):
        w = conv_out(w, k, k // 2, 1)
        n_par = k * k * c * o + 2 * o
        rows.append((f"Conv{i}+BN+ReLU", f"{k}x{k}, {o} filtros, pad {k // 2}", (o, w, w), n_par))
        if pool:
            w = conv_out(w, 2, 0, 2)
            rows.append(("MaxPool", "2x2, passo 2", (o, w, w), 0))
        block_shapes.append((o, w, w))
        c = o
    rows.append(("GlobalAvgPool", "", (c, 1, 1), 0))
    rows.append(("FC1+ReLU", f"{c}->{hidden}", (hidden,), c * hidden + hidden))
    rows.append(("FC2 (logit)", f"{hidden}->1", (1,), hidden + 1))
    return rows, block_shapes


def verify_cnn_dims(model, in_channels=1, input_size=256, hidden=64):
    """Confere a tabela feita a mao com o forward real e com contagem de parametros."""
    rows, expected = cnn_layer_table(in_channels, input_size, hidden)
    got = []
    hooks = [b.register_forward_hook(lambda m, i, o: got.append(tuple(o.shape[1:])))
             for b in model.blocks]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_channels, input_size, input_size))
    model.train(was_training)
    for h in hooks:
        h.remove()
    ok_shape = got == expected
    hand = sum(r[3] for r in rows)
    real = count_params(model)
    return {"ok": ok_shape and hand == real, "shapes_ok": ok_shape,
            "params_hand": hand, "params_real": real, "rows": rows}


def _torchvision_resnet(name, pretrained):
    from torchvision import models
    ctor = getattr(models, name)
    if pretrained:
        try:
            weights = getattr(models, f"{name.capitalize().replace('net', 'Net')}_Weights").IMAGENET1K_V1
        except AttributeError:
            weights = None
        try:
            return ctor(weights=weights)
        except TypeError:
            return ctor(pretrained=True)
    try:
        return ctor(weights=None)
    except TypeError:
        return ctor(pretrained=False)


class PCResNet50(_FreezeMixin, nn.Module):
    """ResNet50 com entrada de 3 canais (fusao: pre/precoce/tardio; fase unica:
    mesma fase replicada nos 3 canais) e cabeca densa 2048 -> hidden -> 1."""

    def __init__(self, in_channels=3, dropout=0.3, head_hidden=256, pretrained=False, **kw):
        super().__init__()
        if in_channels != 3:
            raise ValueError("PCResNet50 espera 3 canais (replique a fase unica).")
        base = _torchvision_resnet("resnet50", pretrained)
        self.norm = _Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = _make_head(base.fc.in_features, head_hidden, dropout)
        self.gradcam_layers = [self.layer4]

    def _backbone_parts(self):
        return [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]

    def _last_block_parts(self):
        return [self.layer4]

    def forward(self, x):
        x = self.norm(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))


def _gray_resnet18(pretrained):
    """ResNet18 com entrada de 1 canal: pesos da 1a conv = soma dos 3 canais
    pre-treinados (mantem a resposta a intensidade media). fc removida."""
    base = _torchvision_resnet("resnet18", pretrained)
    old = base.conv1
    new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
    base.conv1 = new
    base.fc = nn.Identity()
    return base


class PCMultiBranchResNet18(_FreezeMixin, nn.Module):
    """Um ResNet18 INDEPENDENTE por fase (pesos nao compartilhados); as 3 representacoes
    de 512 dimensoes sao concatenadas (1536) e passam pela cabeca densa.
    in_channels=3 -> 3 ramos (fusao);  in_channels=1 -> 1 ramo (fase unica)."""

    def __init__(self, in_channels=3, dropout=0.4, head_hidden=128, pretrained=False, **kw):
        super().__init__()
        self.n_branches = in_channels
        self.norm = _Normalize((GRAY_MEAN,) * in_channels, (GRAY_STD,) * in_channels)
        self.branches = nn.ModuleList([_gray_resnet18(pretrained) for _ in range(in_channels)])
        self.head = _make_head(512 * in_channels, head_hidden, dropout)
        self.gradcam_layers = [b.layer4 for b in self.branches]

    def _backbone_parts(self):
        return list(self.branches)

    def _last_block_parts(self):
        return [b.layer4 for b in self.branches]

    def forward(self, x):
        x = self.norm(x)
        feats = [b(x[:, i:i + 1]) for i, b in enumerate(self.branches)]
        return self.head(torch.cat(feats, dim=1))


def model_in_channels(arch, mode):
    """Canais de entrada do modelo para cada (arquitetura, modo)."""
    if arch == "resnet50":
        return 3
    return 3 if mode == "fusion" else 1


def build_cnn(in_channels=1, dropout=0.5, hidden=64, **kw):
    return PCCNN(in_channels=in_channels, dropout=dropout, hidden=hidden)


def build_resnet50(in_channels=3, dropout=0.3, head_hidden=256, **kw):
    return PCResNet50(in_channels, dropout, head_hidden, pretrained=False)


def build_three_branch(in_channels=3, dropout=0.4, head_hidden=128, **kw):
    return PCMultiBranchResNet18(in_channels, dropout, head_hidden, pretrained=False)


register_arch("cnn", build_cnn)
register_arch("resnet50", build_resnet50)
register_arch("three_branch", build_three_branch)

BUILDERS = {"cnn": build_cnn, "resnet50": build_resnet50, "three_branch": build_three_branch}


def make_model(arch, mode, pretrained=True, dropout=None, hidden=None):
    """Instancia a rede de treino. hidden = largura da camada densa (None = padrao)."""
    cin = model_in_channels(arch, mode)
    kw = {}
    if dropout is not None:
        kw["dropout"] = dropout
    if arch == "cnn":
        if hidden is not None:
            kw["hidden"] = hidden
        return PCCNN(in_channels=cin, **kw)
    if hidden is not None:
        kw["head_hidden"] = hidden
    cls = PCResNet50 if arch == "resnet50" else PCMultiBranchResNet18
    return cls(in_channels=cin, pretrained=pretrained, **kw)


def checkpoint_dict(model, arch, mode, threshold=0.5, input_size=256, extra=None):
    """Dicionario no formato do contrato da Pessoa A (torch.save(dict, caminho))."""
    d = {"arch": arch, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "mode": mode, "in_channels": model_in_channels(arch, mode), "input_size": input_size,
         "output_kind": "logit", "imagenet_norm": False, "threshold": float(threshold),
         "norm": "percentile"}
    d.update(extra or {})
    return d
