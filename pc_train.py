"""
PESSOA C - Treino, metricas, bateria de experimentos e avaliacao no teste.

[Nome, matricula, curso e campus dos componentes do grupo: preencher na integracao]

Uso (sempre dentro da pasta do projeto):
  python pc_train.py cache   --data_dir DATA --csv metadata.csv [--splits_csv splits.csv]
  python pc_train.py check                          # dimensoes da CNN + forward das 3 redes
  python pc_train.py smoke                          # treino-relampago em dados sinteticos
  python pc_train.py train   --arch resnet50 --mode fusion --train_slices tumor
  python pc_train.py battery --seeds 0               # 3 modelos x 4 configs x 3 cortes (retomavel)
  python pc_train.py report                          # tabela principal, curvas e fusao x fase unica
  python pc_train.py test    --ckpt ckpt_pc/<nome>.pt   # TESTE: uma unica vez por modelo/estrategia

Pesos (.pt) vao para --ckpt_dir (FORA do ZIP). Resultados (.json/.csv) sao pequenos.
Treino em fatias 2D; a predicao do paciente agrega os cortes (pc_data.aggregate).
"""
import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import pc_data
import pc_models
from pa_metrics import binary_metrics

MAIN_STRATEGY = {"central": "central", "max_area": "max_area", "tumor": "tumor_wmean"}
PHASE_IDX = {"pre": 0, "early": 1, "late": 2}
METRIC_KEYS = ["sensibilidade", "especificidade", "precisao", "acuracia", "f1", "auc"]


@dataclass
class Exp:
    arch: str = "cnn"
    mode: str = "fusion"
    train_slices: str = "central"
    seed: int = 0
    batch: int = 16
    epochs: int = 60
    lr: float = 1e-3
    epochs_head: int = 15
    lr_head: float = 1e-3
    epochs_ft: int = 15
    lr_ft: float = 1e-4
    finetune: bool = True
    wd: float = 1e-4
    dropout: Optional[float] = None
    hidden: Optional[int] = None
    patience: int = 10
    loss: str = "wbce"
    augment: bool = True
    slices_per_epoch: int = 6
    select: str = "auc"
    thr_method: str = "f1"
    input_size: int = 256
    pretrained: bool = True
    tag: str = ""

    @property
    def phase(self):
        return "fusion" if self.mode == "fusion" else self.mode.split(":")[1]

    @property
    def name(self):
        return f"{self.arch}_{self.phase}_{self.train_slices}_s{self.seed}" + (f"_{self.tag}" if self.tag else "")

    @property
    def main_strategy(self):
        return MAIN_STRATEGY[self.train_slices]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mode_channels(mode):
    return [0, 1, 2] if mode == "fusion" else [PHASE_IDX[mode.split(":")[1]]]


def load_slices(p: pc_data.PatientData, zs, chans) -> np.ndarray:
    """(len(zs), len(chans), X, Y) float32 de um paciente."""
    return np.asarray(p.vol[list(zs)])[:, chans].astype(np.float32)


def to_input(x_np, in_ch, size, device):
    """Identico a pa_models.build_input (sem a normalizacao ImageNet, que ja esta no modelo)."""
    t = torch.from_numpy(x_np)
    if t.shape[1] == 1 and in_ch == 3:
        t = t.repeat(1, 3, 1, 1)
    elif t.shape[1] != in_ch:
        raise ValueError(f"Modelo espera {in_ch} canais, entrada tem {t.shape[1]}")
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.to(device)


def augment_batch(x, rng):
    """Item (b): espelhamento (50%) x rotacao em {-20,-10,0,10,20} graus = 10 variacoes,
    sorteadas por imagem a cada epoca. Mesma transformacao em TODOS os canais (fases).
    Rotacao bilinear em torno do centro, fundo = 0 (preto), tamanho mantido."""
    B = x.shape[0]
    flip = torch.from_numpy(rng.random(B) < 0.5).to(x.device)
    x = torch.where(flip.view(B, 1, 1, 1), x.flip(-1), x)
    ang = torch.from_numpy(rng.choice(pc_data.AUG_ANGLES, B).astype(np.float32)).to(x.device) * math.pi / 180.0
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = c, -s, s, c
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class FocalLoss(nn.Module):
    def __init__(self, alpha_pos, gamma=2.0):
        super().__init__()
        self.a, self.g = alpha_pos, gamma

    def forward(self, logit, y):
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        pt = torch.exp(-bce)
        a = y * self.a + (1 - y) * (1 - self.a)
        return (a * (1 - pt) ** self.g * bce).mean()


def make_loss(kind, n_pos, n_neg, device):
    if kind == "bce":
        return nn.BCEWithLogitsLoss()
    if kind == "wbce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=device))
    if kind == "focal":
        return FocalLoss(alpha_pos=n_neg / max(n_pos + n_neg, 1))
    raise ValueError(kind)


def epoch_samples(pats, mode, k, rng):
    s = []
    for pi, p in enumerate(pats):
        zs = pc_data.training_slices(p.areas, mode)
        if mode == "tumor" and k and len(zs) > k:
            zs = rng.choice(zs, k, replace=False).tolist()
        s += [(pi, int(z)) for z in zs]
    rng.shuffle(s)
    return s


def autocast_ctx(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)


@torch.no_grad()
def predict_logits(model, p, zs, chans, in_ch, size, device, amp, bs=32):
    """Logits (len(zs),) dos cortes zs de um paciente."""
    model.eval()
    out = []
    for i in range(0, len(zs), bs):
        x = to_input(load_slices(p, zs[i:i + bs], chans), in_ch, size, device)
        with autocast_ctx(device, amp):
            lg = model(x)
        out.append(lg.float().reshape(len(x), -1)[:, 0].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def evaluate_main(model, pats, exp, device, amp, crit=None):
    """Validacao por epoca: probabilidade do paciente na estrategia 'natural' do treino
    + loss nos mesmos cortes. Retorna (val_loss, val_auc, y, probs)."""
    chans, in_ch = mode_channels(exp.mode), pc_models.model_in_channels(exp.arch, exp.mode)
    strat = exp.main_strategy
    probs, ys, all_lg, all_y = [], [], [], []
    for p in pats:
        zs = pc_data.slices_needed(p.areas, strat)
        lg = predict_logits(model, p, zs, chans, in_ch, exp.input_size, device, amp)
        per = np.zeros(p.n_slices); per[zs] = 1.0 / (1.0 + np.exp(-lg))
        probs.append(pc_data.aggregate(per, p.areas, strat)); ys.append(p.label)
        all_lg.append(lg); all_y.append(np.full(len(lg), p.label, np.float32))
    vloss = float("nan")
    if crit is not None and all_lg:
        vloss = float(crit(torch.from_numpy(np.concatenate(all_lg)).float().to(device),
                           torch.from_numpy(np.concatenate(all_y)).to(device)).item())
    auc = binary_metrics(ys, probs)["auc"]
    return vloss, auc, np.array(ys), np.array(probs)


def predict_all_strategies(model, pats, exp, device, amp):
    """Probabilidade por paciente para as 5 estrategias de corte (1 inferencia por corte)."""
    chans, in_ch = mode_channels(exp.mode), pc_models.model_in_channels(exp.arch, exp.mode)
    out = {s: [] for s in pc_data.STRATEGIES}
    for p in pats:
        per = 1.0 / (1.0 + np.exp(-predict_logits(model, p, list(range(p.n_slices)), chans, in_ch,
                                                  exp.input_size, device, amp)))
        for s in pc_data.STRATEGIES:
            out[s].append(pc_data.aggregate(per, p.areas, s))
    return {s: np.array(v) for s, v in out.items()}


def make_scaler(amp):
    try:
        return torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp)


def plan_stages(model, exp):
    """CNN: 1 etapa (tudo). Pre-treinadas: (1) so a cabeca densa, (2) + ultimo bloco (layer4)
    com learning rate baixo."""
    if exp.arch == "cnn":
        return [{"name": "tudo", "epochs": exp.epochs, "unfreeze": False, "lr": exp.lr}]
    st = [{"name": "cabeca", "epochs": exp.epochs_head, "unfreeze": False, "lr": exp.lr_head}]
    if exp.finetune and exp.epochs_ft > 0:
        st.append({"name": "layer4", "epochs": exp.epochs_ft, "unfreeze": True, "lr": exp.lr_ft})
    return st


def train_model(exp: Exp, train_pats, val_pats, device, log=print):
    set_seed(exp.seed)
    rng = np.random.default_rng(exp.seed)
    amp = device.type == "cuda"
    model = pc_models.make_model(exp.arch, exp.mode, pretrained=exp.pretrained, dropout=exp.dropout, hidden=exp.hidden).to(device)
    chans, in_ch = mode_channels(exp.mode), pc_models.model_in_channels(exp.arch, exp.mode)
    pretrained_arch = exp.arch != "cnn"
    if pretrained_arch:
        model.freeze_backbone()

    s0 = epoch_samples(train_pats, exp.train_slices, exp.slices_per_epoch, np.random.default_rng(exp.seed))
    n_pos = sum(train_pats[pi].label for pi, _ in s0); n_neg = len(s0) - n_pos
    crit = make_loss(exp.loss, n_pos, n_neg, device)
    log(f"  amostras/epoca={len(s0)}  pCR={n_pos}  nao-pCR={n_neg}  perda={exp.loss}"
        f"  pos_weight={n_neg / max(n_pos, 1):.2f}")

    hist = {"train_loss": [], "val_loss": [], "val_auc": [], "stage_start": []}
    best = {"score": -1e9, "vloss": 1e9, "state": None, "epoch": 0}
    scaler = make_scaler(amp)
    t0, ep = time.time(), 0
    for stage in plan_stages(model, exp):
        if stage["unfreeze"]:
            if best["state"] is not None:
                model.load_state_dict(best["state"])
            model.unfreeze_last_block()
        params = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": params, "lr": stage["lr"]}]
        opt = torch.optim.AdamW(groups, weight_decay=exp.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(stage["epochs"], 1),
                                                           eta_min=stage["lr"] * 0.05)
        hist["stage_start"].append(ep + 1)
        bad = 0
        for _ in range(stage["epochs"]):
            ep += 1
            model.train()
            samples = epoch_samples(train_pats, exp.train_slices, exp.slices_per_epoch, rng)
            tot, cnt = 0.0, 0
            for i in range(0, len(samples), exp.batch):
                chunk = samples[i:i + exp.batch]
                if len(chunk) < 2:
                    continue
                x = np.stack([np.asarray(train_pats[pi].vol[z])[chans] for pi, z in chunk]).astype(np.float32)
                y = torch.tensor([train_pats[pi].label for pi, _ in chunk], dtype=torch.float32, device=device)
                xt = to_input(x, in_ch, exp.input_size, device)
                if exp.augment:
                    xt = augment_batch(xt, rng)
                opt.zero_grad(set_to_none=True)
                with autocast_ctx(device, amp):
                    lg = model(xt)
                loss = crit(lg.float().reshape(-1), y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update()
                tot += loss.item() * len(chunk); cnt += len(chunk)
            sched.step()
            vloss, vauc, _, _ = evaluate_main(model, val_pats, exp, device, amp, crit)
            tl = tot / max(cnt, 1)
            hist["train_loss"].append(tl); hist["val_loss"].append(vloss); hist["val_auc"].append(vauc)
            score = -vloss if exp.select == "loss" else (vauc if not math.isnan(vauc) else 0.5)
            improved = score > best["score"] + 1e-6 or (abs(score - best["score"]) <= 1e-6 and vloss < best["vloss"])
            if improved:
                best.update(score=score, vloss=vloss, epoch=ep,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                bad = 0
            else:
                bad += 1
            log(f"  [{stage['name']}] ep {ep:03d} loss={tl:.4f} val_loss={vloss:.4f} val_auc={vauc:.3f}"
                f"{' *' if improved else ''}")
            if bad >= exp.patience:
                log(f"  early stopping (paciencia {exp.patience})")
                break
    train_time = time.time() - t0
    model.load_state_dict(best["state"])
    model.eval()
    hist.update(best_epoch=best["epoch"], epochs_run=ep, train_time_s=train_time)
    return model, hist


LOG_COLS = (["exp", "arch", "phase", "train_slices", "eval_strategy", "seed", "loss", "augment", "lr", "lr_ft",
             "wd", "dropout", "hidden", "batch", "epochs_run", "best_epoch", "train_time_s", "n_params",
             "thr_val"] + [f"m05_{k}" for k in METRIC_KEYS] + [f"mt_{k}" for k in METRIC_KEYS] + ["ckpt"])


def append_log(rows, path):
    df = pd.DataFrame(rows).reindex(columns=LOG_COLS)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_experiment(exp: Exp, ctx, force=False, log=print):
    names = ctx["paths"]
    res_json = os.path.join(names["results"], f"{exp.name}.json")
    if os.path.exists(res_json) and not force:
        log(f"[pulado] {exp.name} (ja existe)")
        return None
    log(f"\n=== {exp.name} | modelo={exp.arch} fases={exp.phase} cortes={exp.train_slices} ===")
    device = ctx["device"]
    model, hist = train_model(exp, ctx["train"], ctx["val"], device, log)
    amp = device.type == "cuda"
    probs = predict_all_strategies(model, ctx["val"], exp, device, amp)
    y = np.array([p.label for p in ctx["val"]])
    thr_main = pc_data.choose_threshold(y, probs[exp.main_strategy], exp.thr_method)

    rows, all_metrics = [], {}
    n_params = pc_models.count_params(model)
    for s in pc_data.STRATEGIES:
        thr_s = pc_data.choose_threshold(y, probs[s], exp.thr_method)
        m05, mt = binary_metrics(y, probs[s], 0.5), binary_metrics(y, probs[s], thr_s)
        all_metrics[s] = {"thr": thr_s, "m05": m05, "mt": mt}
        row = {"exp": exp.name, "arch": exp.arch, "phase": exp.phase, "train_slices": exp.train_slices,
               "eval_strategy": s, "seed": exp.seed, "loss": exp.loss, "augment": int(exp.augment),
               "lr": exp.lr if exp.arch == "cnn" else exp.lr_head, "lr_ft": exp.lr_ft if exp.arch != "cnn" else "",
               "wd": exp.wd, "dropout": exp.dropout, "hidden": exp.hidden, "batch": exp.batch,
               "epochs_run": hist["epochs_run"], "best_epoch": hist["best_epoch"],
               "train_time_s": round(hist["train_time_s"], 1), "n_params": n_params, "thr_val": thr_s,
               "ckpt": os.path.join(names["ckpt"], f"{exp.name}.pt")}
        row.update({f"m05_{k}": m05[k] for k in METRIC_KEYS})
        row.update({f"mt_{k}": mt[k] for k in METRIC_KEYS})
        rows.append(row)

    hp = {"dropout": exp.dropout if exp.dropout is not None else
          {"cnn": 0.5, "resnet50": 0.3, "three_branch": 0.4}[exp.arch]}
    default_h = {"cnn": 64, "resnet50": 256, "three_branch": 128}[exp.arch]
    hp["hidden" if exp.arch == "cnn" else "head_hidden"] = exp.hidden or default_h
    os.makedirs(names["ckpt"], exist_ok=True)
    torch.save(pc_models.checkpoint_dict(model, exp.arch, exp.mode, thr_main, exp.input_size,
                                         extra={**hp, "strategy": exp.main_strategy, "train_slices": exp.train_slices,
                                                "exp_name": exp.name}), os.path.join(names["ckpt"], f"{exp.name}.pt"))
    mm = all_metrics[exp.main_strategy]["mt"]
    with open(res_json, "w", encoding="utf-8") as f:
        json.dump({"name": f"{exp.arch} | {exp.phase} | {exp.train_slices}",
                   "train_loss": hist["train_loss"], "val_loss": hist["val_loss"], "val_auc": hist["val_auc"],
                   "train_time_s": hist["train_time_s"], "metrics": {k: mm[k] for k in METRIC_KEYS},
                   "notes": f"VALIDACAO; limiar={thr_main:.3f} ({exp.thr_method}); estrategia={exp.main_strategy}; "
                            f"melhor epoca={hist['best_epoch']}; params={n_params}",
                   "stage_start": hist["stage_start"], "exp": asdict(exp),
                   "all_strategies": {s: {"thr": v["thr"], "auc": v["mt"]["auc"], "f1_thr05": v["m05"]["f1"],
                                          "f1_thr": v["mt"]["f1"]} for s, v in all_metrics.items()}},
                  f, indent=1, ensure_ascii=False)
    append_log(rows, names["log"])
    log(f"  >> VAL ({exp.main_strategy}): AUC={mm['auc']:.3f} F1={mm['f1']:.3f} sens={mm['sensibilidade']:.3f} "
        f"espec={mm['especificidade']:.3f} thr={thr_main:.2f} | tempo={hist['train_time_s']:.0f}s")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def make_ctx(args, smoke=False):
    cache = args.cache
    train, val = pc_data.load_cache(cache, "train"), pc_data.load_cache(cache, "val")
    pc_data.check_no_leakage({"train": train, "val": val, "test": pc_data.load_cache(cache, "test")})
    paths = {"results": args.results, "ckpt": args.ckpt_dir, "log": os.path.join(args.results, "log_experimentos.csv")}
    os.makedirs(args.results, exist_ok=True)
    print(f"Dispositivo: {get_torch_device()} | treino={len(train)} val={len(val)} pacientes | "
          f"pCR treino={sum(p.label for p in train)}/{len(train)}")
    return {"train": train, "val": val, "paths": paths, "device": get_torch_device()}


def exp_from_args(a, arch, mode, ts, seed):
    kw = {k: getattr(a, k) for k in ("batch", "wd", "patience", "loss", "slices_per_epoch", "select", "thr_method",
                                     "input_size", "tag") if hasattr(a, k)}
    e = Exp(arch=arch, mode=mode, train_slices=ts, seed=seed, augment=not a.no_aug, finetune=not a.no_finetune,
            dropout=a.dropout, hidden=a.hidden, **kw)
    for k in ("epochs", "lr", "epochs_head", "lr_head", "epochs_ft", "lr_ft"):
        if getattr(a, k, None) is not None:
            setattr(e, k, getattr(a, k))
    return e


def cmd_cache(a):
    pc_data.build_cache(a.data_dir, a.csv, a.cache, a.splits_csv)


def cmd_check(a):
    r = pc_models.verify_cnn_dims(pc_models.PCCNN(1), 1, 256, 64)
    print("CNN propria (entrada 1x256x256) - dimensoes feitas a mao (W=floor((W-K+2P)/S)+1):")
    for name, desc, shp, par in r["rows"]:
        print(f"  {name:<16}{desc:<28}saida={str(shp):<16}parametros={par}")
    print(f"  total a mao={r['params_hand']}  total real={r['params_real']}  formas_ok={r['shapes_ok']}  -> "
          f"{'OK' if r['ok'] else 'DIVERGE'}")
    for arch in ("cnn", "resnet50", "three_branch"):
        for mode in ("single:pre", "fusion"):
            m = pc_models.make_model(arch, mode, pretrained=False).eval()
            cin = pc_models.model_in_channels(arch, mode)
            with torch.no_grad():
                o = m(torch.rand(2, cin, 256, 256))
            print(f"  {arch:<13}{mode:<12} in={cin} out={tuple(o.shape)} params={pc_models.count_params(m):,}")
    print(pc_models.make_model("cnn", "fusion", False))


def cmd_smoke(a):
    d = "_smoke_cache"
    pc_data.make_synthetic_cache(d)
    a.cache, a.results, a.ckpt_dir = d, "_smoke_results", "_smoke_ckpt"
    ctx = make_ctx(a)
    for arch in ("cnn", "resnet50", "three_branch"):
        for mode, ts in (("fusion", "tumor"), ("single:late", "central")):
            e = Exp(arch=arch, mode=mode, train_slices=ts, epochs=2, epochs_head=1, epochs_ft=1, batch=8,
                    input_size=64, slices_per_epoch=3, pretrained=False)
            run_experiment(e, ctx, force=True)
    print("\nSMOKE OK: o pipeline completo roda. Apague _smoke_cache/_smoke_results/_smoke_ckpt.")


def cmd_train(a):
    ctx = make_ctx(a)
    for seed in a.seeds:
        run_experiment(exp_from_args(a, a.arch, a.mode, a.train_slices, seed), ctx, force=a.force)


def cmd_battery(a):
    ctx = make_ctx(a)
    todo = [(ar, mo, ts, sd) for ar in a.archs for mo in a.modes for ts in a.train_slices_list for sd in a.seeds]
    print(f"Bateria: {len(todo)} experimentos (retomavel: pula os que ja tem .json)")
    for i, (ar, mo, ts, sd) in enumerate(todo, 1):
        print(f"\n##### {i}/{len(todo)}")
        try:
            run_experiment(exp_from_args(a, ar, mo, ts, sd), ctx, force=a.force)
        except Exception as ex:
            print(f"[ERRO] {ar} {mo} {ts} seed={sd}: {ex}")


def _tex_table(df, cols, caption, label):
    head = " & ".join(cols)
    lines = [r"\begin{table}[h]", r"\centering", r"\small", rf"\begin{{tabular}}{{{'l' * len(cols)}}}", r"\hline",
             head + r" \\", r"\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(r[c]) for c in cols) + r" \\")
    lines += [r"\hline", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log = pd.read_csv(os.path.join(a.results, "log_experimentos.csv"))
    log["natural"] = log.apply(lambda r: r["eval_strategy"] == MAIN_STRATEGY[r["train_slices"]], axis=1)
    keys = ["arch", "phase", "train_slices", "eval_strategy"]
    agg = log.groupby(keys).agg(n_seeds=("seed", "nunique"), **{f"{k}_mean": (f"mt_{k}", "mean") for k in METRIC_KEYS},
                                auc_std=("mt_auc", "std"), f1_std=("mt_f1", "std"),
                                tempo_s=("train_time_s", "mean")).reset_index()
    agg.to_csv(os.path.join(a.results, "tabela_completa.csv"), index=False)
    nat = agg[agg.apply(lambda r: r["eval_strategy"] == MAIN_STRATEGY[r["train_slices"]], axis=1)].copy()
    nat = nat.sort_values(["arch", "phase", "auc_mean"], ascending=[True, True, False])
    nat.round(3).to_csv(os.path.join(a.results, "tabela_principal.csv"), index=False)
    disp = nat.round(3).astype({"n_seeds": int})
    print("\nTABELA PRINCIPAL (validacao, estrategia natural de cada treino; media entre seeds):")
    print(disp[["arch", "phase", "train_slices", "n_seeds", "auc_mean", "f1_mean", "sensibilidade_mean",
                "especificidade_mean", "tempo_s"]].to_string(index=False))
    with open(os.path.join(a.results, "tabela_principal.tex"), "w", encoding="utf-8") as f:
        t = disp.rename(columns={"auc_mean": "AUC", "f1_mean": "F1", "sensibilidade_mean": "Sens.",
                                 "especificidade_mean": "Espec.", "precisao_mean": "Prec.", "acuracia_mean": "Acur."})
        f.write(_tex_table(t, ["arch", "phase", "train_slices", "AUC", "F1", "Sens.", "Espec.", "Prec.", "Acur."],
                           "Desempenho na validacao (limiar escolhido na validacao).", "tab:principal"))
    print("\nHIPOTESE - 3 fases (fusao) x melhor fase unica (AUC na validacao, melhor corte de cada grupo):")
    for arch, g in nat.groupby("arch"):
        best = g.sort_values("auc_mean", ascending=False).groupby("phase").head(1).set_index("phase")["auc_mean"]
        singles = best.drop("fusion", errors="ignore")
        if "fusion" in best and len(singles):
            print(f"  {arch:<13} fusao={best['fusion']:.3f}  melhor fase unica={singles.max():.3f} "
                  f"({singles.idxmax()})  diferenca={best['fusion'] - singles.max():+.3f}")

    best_rows = log[log.natural].sort_values("mt_auc", ascending=False).groupby("arch").head(1)
    fig, axs = plt.subplots(1, max(len(best_rows), 1), figsize=(5 * max(len(best_rows), 1), 3.6), squeeze=False)
    for ax, (_, r) in zip(axs[0], best_rows.iterrows()):
        j = json.load(open(os.path.join(a.results, f"{r['exp']}.json"), encoding="utf-8"))
        ax.plot(j["train_loss"], label="treino"); ax.plot(j["val_loss"], label="validacao")
        for s in j.get("stage_start", [])[1:]:
            ax.axvline(s - 1, ls="--", c="gray", lw=0.8)
        ax.set_title(r["exp"], fontsize=8); ax.set_xlabel("epoca"); ax.set_ylabel("loss"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.results, "curvas_loss.png"), dpi=150)
    print(f"\nSalvos em {a.results}: tabela_principal.csv/.tex, tabela_completa.csv, curvas_loss.png")


def bootstrap_ci(y, p, thr, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    aucs, f1s = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = binary_metrics(y[i], p[i], thr)
        aucs.append(m["auc"]); f1s.append(m["f1"])
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float("nan"),) * 2
    return {"auc_ci95": q(aucs), "f1_ci95": q(f1s)}


def load_ckpt_model(path, device):
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=device)
    model = pc_models.BUILDERS[ck["arch"]](**{k: v for k, v in ck.items() if k != "state_dict"})
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def cmd_test(a):
    """Avaliacao no TESTE. Cada (checkpoint, estrategia) so pode ser avaliado UMA vez."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    device = get_torch_device()
    model, ck = load_ckpt_model(a.ckpt, device)
    strat = a.strategy or ck.get("strategy", "mean")
    key = f"{os.path.basename(a.ckpt)}|{strat}"
    lock_path = os.path.join(a.results, "TESTE_avaliacoes.json")
    lock = json.load(open(lock_path)) if os.path.exists(lock_path) else {}
    if key in lock and not a.force:
        raise SystemExit(f"BLOQUEADO: {key} ja foi avaliado no teste em {lock[key]}. "
                         f"O enunciado permite uma unica avaliacao (use --force so para depurar).")
    test = pc_data.load_cache(a.cache, "test")
    exp = Exp(arch=ck["arch"], mode=ck["mode"], input_size=ck["input_size"])
    probs = predict_all_strategies(model, test, exp, device, device.type == "cuda")[strat]
    y = np.array([p.label for p in test])
    thr = a.threshold if a.threshold is not None else ck["threshold"]
    m = binary_metrics(y, probs, thr)
    ci = bootstrap_ci(y, probs, thr)
    tag = f"{ck.get('exp_name', os.path.basename(a.ckpt)[:-3])}_{strat}"
    print(f"\nTESTE {tag} (n={len(y)}, limiar={thr:.3f})")
    print(f"  Sens={m['sensibilidade']:.3f} Espec={m['especificidade']:.3f} Prec={m['precisao']:.3f} "
          f"Acur={m['acuracia']:.3f} F1={m['f1']:.3f} AUC={m['auc']:.3f}  (IC95% AUC {ci['auc_ci95'][0]:.2f}-{ci['auc_ci95'][1]:.2f})")
    print(f"  TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
    errs = pd.DataFrame({"pid": [p.pid for p in test], "label": y, "prob": probs, "pred": (probs >= thr).astype(int)})
    errs["erro"] = np.where(errs.pred == errs.label, "",
                            np.where(errs.pred == 1, "FP (previu pCR, nao houve)", "FN (previu nao-pCR, houve pCR)"))
    errs.sort_values("prob").to_csv(os.path.join(a.results, f"teste_{tag}_predicoes.csv"), index=False)
    print("  Erros:", errs[errs.erro != ""][["pid", "label", "prob", "erro"]].round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["nao-pCR", "pCR"]); ax.set_yticks([0, 1], ["nao-pCR", "pCR"])
    ax.set_xlabel("predito"); ax.set_ylabel("real"); ax.set_title(tag, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.results, f"teste_{tag}_confusao.png"), dpi=150)
    src = os.path.join(a.results, f"{ck.get('exp_name', '')}.json")
    base = json.load(open(src, encoding="utf-8")) if os.path.exists(src) else {}
    out = {"name": f"TESTE {tag}", "train_loss": base.get("train_loss", []), "val_loss": base.get("val_loss", []),
           "train_time_s": base.get("train_time_s"), "metrics": {k: m[k] for k in METRIC_KEYS},
           "notes": f"TESTE; limiar={thr:.3f}; estrategia={strat}; IC95 AUC={ci['auc_ci95']}", "ci": ci,
           "confusao": {k: m[k] for k in ("tp", "tn", "fp", "fn")}}
    json.dump(out, open(os.path.join(a.results, f"teste_{tag}.json"), "w", encoding="utf-8"), indent=1)
    lock[key] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(lock, open(lock_path, "w"), indent=1)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--cache", default="cache_pc")
        sp.add_argument("--results", default="results_pc")
        sp.add_argument("--ckpt_dir", default="ckpt_pc")

    def hp(sp):
        sp.add_argument("--epochs", type=int); sp.add_argument("--lr", type=float)
        sp.add_argument("--epochs_head", type=int); sp.add_argument("--lr_head", type=float)
        sp.add_argument("--epochs_ft", type=int); sp.add_argument("--lr_ft", type=float)
        sp.add_argument("--batch", type=int, default=16); sp.add_argument("--wd", type=float, default=1e-4)
        sp.add_argument("--dropout", type=float); sp.add_argument("--hidden", type=int)
        sp.add_argument("--patience", type=int, default=10)
        sp.add_argument("--loss", default="wbce", choices=["bce", "wbce", "focal"])
        sp.add_argument("--slices_per_epoch", type=int, default=6)
        sp.add_argument("--select", default="auc", choices=["auc", "loss"])
        sp.add_argument("--thr_method", default="f1", choices=["f1", "youden"])
        sp.add_argument("--input_size", type=int, default=256)
        sp.add_argument("--no_aug", action="store_true"); sp.add_argument("--no_finetune", action="store_true")
        sp.add_argument("--tag", default=""); sp.add_argument("--force", action="store_true")
        sp.add_argument("--seeds", type=int, nargs="+", default=[0])

    s = sub.add_parser("cache"); common(s)
    s.add_argument("--data_dir", required=True); s.add_argument("--csv", required=True)
    s.add_argument("--splits_csv"); s.set_defaults(fn=cmd_cache)
    s = sub.add_parser("check"); common(s); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("smoke"); common(s); s.set_defaults(fn=cmd_smoke, no_aug=False, no_finetune=False)
    s = sub.add_parser("train"); common(s); hp(s)
    s.add_argument("--arch", required=True, choices=["cnn", "resnet50", "three_branch"])
    s.add_argument("--mode", default="fusion", choices=["fusion", "single:pre", "single:early", "single:late"])
    s.add_argument("--train_slices", default="central", choices=["central", "max_area", "tumor"])
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("battery"); common(s); hp(s)
    s.add_argument("--archs", nargs="+", default=["cnn", "resnet50", "three_branch"])
    s.add_argument("--modes", nargs="+", default=["single:pre", "single:early", "single:late", "fusion"])
    s.add_argument("--train_slices_list", nargs="+", default=["central", "max_area", "tumor"])
    s.set_defaults(fn=cmd_battery)
    s = sub.add_parser("report"); common(s); s.set_defaults(fn=cmd_report)
    s = sub.add_parser("test"); common(s)
    s.add_argument("--ckpt", required=True); s.add_argument("--strategy", choices=pc_data.STRATEGIES)
    s.add_argument("--threshold", type=float); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_test)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
