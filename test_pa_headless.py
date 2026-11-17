import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_synthetic_data import make
from pa_augment import augment_variants, augment_mask_like, balance_plan
from pa_gradcam import heat_in_mask_metrics, _norm01, summarize_heat_metrics
from pa_metrics import auc_roc, binary_metrics
from pa_nii_io import DatasetIndex, load_case, orient_slice
from pa_slices import predict_volume, representative_slice, max_area_slice, tumor_area_per_slice


def ok(name):
    print("  OK ", name)


def main():
    tmp = tempfile.mkdtemp()
    make(tmp, n=18, size=64, nz=8)


    idx = DatasetIndex(tmp)
    assert len(idx.cases) == 18 and len(idx.pids()) == 18, idx.summary()
    assert idx.csv_info["label"] == "pcr" and idx.csv_info["split"] == "split"
    assert set(idx.cases["P001"].phases) == {"pre", "early", "late"} and idx.cases["P001"].mask
    assert idx.cases["P001"].label == 1 and idx.cases["P002"].label == 0
    assert len(idx.pids("train")) == 11 and len(idx.pids("val")) == 4 and len(idx.pids("test")) == 3
    ok("DatasetIndex: pacientes, fases, mascara, rotulo, splits")

    lc = load_case(idx.cases["P001"])
    assert lc.n_slices == 8 and lc.vols["early"].shape == (64, 64, 8)
    assert 0 <= lc.vols["pre"].min() and lc.vols["pre"].max() <= 1
    assert lc.mask.shape == lc.vols["pre"].shape and set(np.unique(lc.mask)) <= {0.0, 1.0}
    assert lc.info["pre"].axcodes == "RAS"
    ok("load_case: shape, normalizacao [0,1], mascara binaria, axcodes")

    img, origin = orient_slice(np.arange(12).reshape(3, 4), "T_lower")
    assert img.shape == (4, 3) and origin == "lower"
    assert orient_slice(np.zeros((3, 4)), "T_lower", flip_lr=True)[0].shape == (4, 3)
    ok("orient_slice")


    sl = lc.vols["early"][:, :, 4]
    vs = augment_variants(sl)
    assert len(vs) == 10 and all(v[1].shape == sl.shape for v in vs)
    assert np.allclose(vs[2][1], sl)
    assert np.allclose(vs[7][1], sl[:, ::-1])
    assert not np.allclose(vs[0][1], sl)
    m2 = augment_mask_like(lc.mask[:, :, 4], 10, True)
    assert set(np.unique(m2)) <= {0.0, 1.0}
    ok("augment_variants: 10 variacoes, identidade e espelho corretos")


    p = balance_plan({0: 120, 1: 40})
    assert abs(p["ratio"] - 3.0) < 1e-9 and p["relevant"]
    assert p["variants_per_class"] == {0: 10, 1: 30}
    assert p["after_prop"][0] == p["after_prop"][1] == 1200
    assert abs(p["pos_weight"] - 3.0) < 1e-9 and abs(p["class_weights"][1] - 2.0) < 1e-9
    assert not balance_plan({0: 50, 1: 45})["relevant"]
    ok("balance_plan")


    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    pr = np.clip(y * 0.3 + rng.normal(0.35, 0.25, 200), 0, 1)
    m = binary_metrics(y, pr, 0.5)
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
    yp = (pr >= 0.5).astype(int)
    assert abs(m["auc"] - roc_auc_score(y, pr)) < 1e-9
    assert abs(m["precisao"] - precision_score(y, yp)) < 1e-9
    assert abs(m["sensibilidade"] - recall_score(y, yp)) < 1e-9
    assert abs(m["f1"] - f1_score(y, yp)) < 1e-9 and abs(m["acuracia"] - accuracy_score(y, yp)) < 1e-9
    assert abs(auc_roc([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9
    ok("metricas e AUC conferem com sklearn")


    areas = tumor_area_per_slice(lc.mask, 8)
    assert areas.shape == (8,) and max_area_slice(lc.mask, 8) == int(np.argmax(areas))
    fake = lambda zs: np.array([0.1 + 0.1 * z for z in zs])
    calls = []
    spy = lambda zs: (calls.append(len(zs)), fake(zs))[1]
    r = predict_volume(spy, 8, lc.mask, "central")
    assert r["slices"] == [4] and abs(r["prob"] - 0.5) < 1e-9 and calls == [1]
    r = predict_volume(fake, 8, lc.mask, "max_area")
    assert r["slices"] == [int(np.argmax(areas))]
    r = predict_volume(fake, 8, lc.mask, "mean")
    assert abs(r["prob"] - np.mean([0.1 + 0.1 * z for z in range(8)])) < 1e-9
    r = predict_volume(fake, 8, lc.mask, "max")
    assert abs(r["prob"] - 0.8) < 1e-9 and representative_slice(r, "max", 8) == 7
    r = predict_volume(fake, 8, lc.mask, "tumor_wmean")
    w = areas / areas.sum()
    assert abs(r["prob"] - float((fake(range(8)) * w).sum())) < 1e-9
    for bad in ("max_area", "tumor_wmean"):
        try:
            predict_volume(fake, 8, None, bad)
            raise SystemExit("deveria exigir mascara: " + bad)
        except ValueError:
            pass
    ok("estrategias: central, maior area, media, maximo, media ponderada (+erro sem mascara)")


    mask = np.zeros((64, 64)); mask[20:30, 20:30] = 1
    cam_in = np.zeros((64, 64)); cam_in[22:28, 22:28] = 1.0
    cam_out = np.zeros((64, 64)); cam_out[45:55, 45:55] = 1.0
    a, b = heat_in_mask_metrics(cam_in, mask), heat_in_mask_metrics(cam_out, mask)
    assert abs(a["heat_fraction"] - 1.0) < 1e-9 and a["peak_in_mask"] == 1.0 and a["top_in_mask"] == 1.0
    assert b["heat_fraction"] == 0.0 and b["peak_in_mask"] == 0.0
    assert abs(a["enrichment"] - 1.0 / (100 / 4096)) < 1e-6
    half = cam_in.copy(); half[45:55, 45:55] = 36 / 100
    assert abs(heat_in_mask_metrics(half, mask)["heat_fraction"] - 36 / 72) < 1e-9
    assert np.isnan(heat_in_mask_metrics(cam_in, np.zeros((64, 64)))["heat_fraction"])
    assert np.isnan(heat_in_mask_metrics(np.zeros((64, 64)), mask)["heat_fraction"])
    s = summarize_heat_metrics([a, b, heat_in_mask_metrics(cam_in, np.zeros((64, 64)))])
    assert abs(s["heat_fraction"] - 0.5) < 1e-9 and s["heat_fraction_n"] == 2
    n01 = _norm01(np.stack([np.arange(4.0).reshape(2, 2), np.zeros((2, 2))]))
    assert n01[0].max() == 1 and n01[0].min() == 0 and n01[1].max() == 0
    ok("heat_in_mask_metrics / _norm01")


    import pa_models
    import pa_folder_eval

    def fake_make_predict_fn(spec, vols, batch_size=16):

        def f(zs):
            return np.array([1.0 - np.clip((vols["early"][:, :, z] > 0.5).mean() / 0.12, 0, 1)
                             for z in zs])
        return f
    pa_models.make_predict_fn = fake_make_predict_fn
    class Spec: threshold = 0.5; phases_used = ["pre", "early", "late"]
    out = pa_folder_eval.classify_folder(Spec(), tmp, "max_area")
    assert len(out["rows"]) == 18 and all(r["erro"] == "" for r in out["rows"])
    mt = out["metrics"]
    assert mt["n"] == 18 and mt["tp"] + mt["tn"] + mt["fp"] + mt["fn"] == 18
    assert mt["auc"] > 0.9, mt
    out_t = pa_folder_eval.classify_folder(Spec(), tmp, "mean", split="test")
    assert len(out_t["rows"]) == 3

    nm = tempfile.mkdtemp()
    for f in os.listdir(tmp):
        if "seg" not in f:
            os.link(os.path.join(tmp, f), os.path.join(nm, f))
    out_nm = pa_folder_eval.classify_folder(Spec(), nm, "max_area")
    assert {r["strategy_used"] for r in out_nm["rows"]} == {"mean"} and out_nm["metrics"]

    nc = tempfile.mkdtemp()
    for f in os.listdir(tmp):
        if f.endswith(".gz"):
            os.link(os.path.join(tmp, f), os.path.join(nc, f))
    out_nc = pa_folder_eval.classify_folder(Spec(), nc, "mean")
    assert len(out_nc["rows"]) == 18 and out_nc["metrics"] is None
    ok("classify_folder: metricas, split, fallback sem mascara, pasta sem CSV")

    test_real_like()
    test_meta_layout()
    print("\nTODOS OS TESTES SEM TORCH PASSARAM")


def test_real_like():
    import nibabel as nib
    from pa_nii_io import detect_z_axis, load_nii
    import pa_config


    base = np.random.default_rng(0).random((32, 32, 5)).astype(np.float32)
    for za, arr in ((2, base), (1, np.transpose(base, (0, 2, 1))), (0, np.transpose(base, (2, 0, 1)))):
        f = os.path.join(tempfile.mkdtemp(), "v.nii.gz")
        nib.save(nib.Nifti1Image(arr, np.eye(4)), f)
        v = load_nii(f)
        assert v.shape == (32, 32, 5) and v.z_axis_original == za, (za, v.shape)
        assert np.allclose(v.data, base)
    assert detect_z_axis((256, 49, 256)) == 1 and detect_z_axis((49, 256, 256)) == 0
    assert detect_z_axis((256, 256, 49)) == 2
    assert detect_z_axis((10, 20, 30)) == 0 and detect_z_axis((30, 20, 10)) == 2
    assert detect_z_axis((10, 20, 30), forced=2) == 2
    ok("eixo z detectado (0, 1 ou 2) e movido para o fim; conteudo preservado")


    for with_cols in (True, False):
        tmp = tempfile.mkdtemp()
        make(tmp, n=20, size=64, nz=8, layout="official", phase_cols=with_cols)
        idx = DatasetIndex(tmp)
        assert idx.csv_info["rows_total"] == 25 and idx.csv_info["rows_used"] == 20, idx.csv_info
        assert len(idx.cases) == 20 and not any(p.startswith("DUKE") for p in idx.cases)
        assert (idx.csv_info["phase_cols"] is not None) == with_cols
        assert len(idx.pids()) == 20

        assert (len(idx.pids("train")), len(idx.pids("test")), len(idx.pids("val"))) == (12, 4, 4)
        assert idx.cases["ISPY1_1017"].split == "test" and idx.cases["ISPY1_1013"].split == "val"

        import pandas as pd
        os.makedirs(os.path.join(tmp, "cache"), exist_ok=True)
        pd.DataFrame({"pid": list(idx.cases), "pCR": 0, "split": "train"}).to_csv(
            os.path.join(tmp, "cache", "splits.csv"), index=False)
        idx2 = DatasetIndex(tmp)
        assert idx2.csv_info["splits_csv"].endswith("splits.csv") and idx2.csv_info["splits_applied"] == 0
        assert (len(idx2.pids("train")), len(idx2.pids("test")), len(idx2.pids("val"))) == (12, 4, 4)
        lc = load_case(idx.cases["ISPY1_1000"])
        assert lc.vols["early"].shape == (64, 64, 8) and lc.mask.shape == (64, 64, 8)
        areas = lc.mask.sum(axis=(0, 1))
        assert int(np.argmax(areas)) in (3, 4, 5), areas
        assert idx.cases["ISPY1_1000"].mask_range() is not None
        zs = np.where(areas > 0)[0]
        assert idx.cases["ISPY1_1000"].mask_range() == (int(zs.min()), int(zs.max()))
    ok("dataset real-like: filtro spy1, test 0/1/2, colunas pre/post_*, _aqc_, eixo z, mask_start/end")


def test_meta_layout():
    import nibabel as nib
    tmp = tempfile.mkdtemp()
    make(tmp, n=20, size=64, nz=8, layout="meta")
    raw = np.asarray(nib.load(os.path.join(tmp, "dce", "ISPY1_1000_spy1_vis1_dce_aqc_0.nii.gz")).dataobj)
    assert raw.shape == (8, 64, 64)
    assert raw.shape[2] // 2 != 4

    idx = DatasetIndex(tmp)
    assert idx.csv_info["path"].endswith("BreastDCEDL_spy1_metadata.csv"), idx.csv_info["path"]
    assert idx.csv_info["rows_used"] == idx.csv_info["rows_total"] == 22
    assert idx.csv_info["splits_csv"].endswith(os.path.join("cache", "splits.csv"))
    assert len(idx.cases) == 20 and len(idx.pids()) == 20
    assert all(c.mask and os.sep + "mask" + os.sep in c.mask for c in idx.cases.values())
    assert (len(idx.pids("train")), len(idx.pids("val")), len(idx.pids("test"))) == (12, 4, 4)
    assert idx.class_counts() == {0: 13, 1: 7}
    for pid in ("ISPY1_1000", "ISPY1_1013", "ISPY1_1017"):
        lc = load_case(idx.cases[pid])
        assert lc.vols["pre"].shape == (64, 64, 8) and lc.info["pre"].z_axis_original == 0
        areas = lc.mask.sum(axis=(0, 1))

        assert int(np.argmax(areas)) == int(idx.cases[pid].clinical["mask_max_plane"]), areas
    from pa_models import _filter_kwargs
    ok_builder = lambda dropout=0.5, hidden=64: None
    cfg = {"arch": "cnn", "dropout": 0.3, "hidden": 32, "threshold": 0.4, "norm": "percentile"}
    assert _filter_kwargs(ok_builder, cfg) == {"dropout": 0.3, "hidden": 32}
    assert _filter_kwargs(lambda **kw: None, cfg) == cfg
    ok("layout da Pessoa B: dce/ + mask/, _aqc_, CSV de metadados, splits.csv, z no eixo 0, mask_max_plane")
    ok("_filter_kwargs: builder so recebe os argumentos que aceita")


if __name__ == "__main__":
    main()
