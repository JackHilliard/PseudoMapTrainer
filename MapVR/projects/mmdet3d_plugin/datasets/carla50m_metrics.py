"""Per-class precision and chamfer distance for the 50m CARLA benchmark.

Kept in its OWN module with only numpy/shapely imports: importing it through
the plugin package pulls mmdet -> scipy, which explodes if a numpy 2.x leaks in
from ~/.local. This module can be imported standalone for testing.
"""
import numpy as np


def _resample(pts, n=100):
    from shapely.geometry import LineString
    L = LineString(np.asarray(pts, dtype=float)[:, :2])
    if L.length < 1e-9:
        return np.repeat(np.asarray(pts, dtype=float)[:1, :2], n, 0)
    return np.array([list(L.interpolate(d).coords)[0]
                     for d in np.linspace(0, L.length, n)])


def precision_and_chamfer(results, gt_annos, class_names, thresholds=(0.5, 1.0, 1.5),
                          chamfer_thr=1.5):
    """Per-class precision/recall/F1 at max-F1, plus matched chamfer distance.

    Returns {f'{cls}_precision_thr_{t}': v, ..., f'{cls}_chamfer': v}.
    """
    gt_by = {g['sample_token']: g for g in gt_annos}
    cache = {}
    for r in results:
        tok = r['sample_token']
        if tok not in gt_by:
            continue
        per = {}
        for ci, _ in enumerate(class_names):
            gv = [v for v in gt_by[tok]['vectors'] if int(v.get('type', 0)) == ci]
            pv = [v for v in r['vectors']
                  if int(v.get('type', 0)) == ci and len(v['pts']) >= 2]
            sc = np.array([float(v['confidence_level']) for v in pv])
            if not pv or not gv:
                per[ci] = (np.zeros((len(pv), max(len(gv), 1))), sc, len(gv))
                continue
            P = np.stack([_resample(v['pts']) for v in pv])
            G = np.stack([_resample(g['pts']) for g in gv])
            d = np.linalg.norm(P[:, None, :, None, :] - G[None, :, None, :, :], axis=-1)
            per[ci] = (0.5 * (d.min(3).mean(2) + d.min(2).mean(2)), sc, len(gv))
        cache[tok] = per

    def voc_ap(rec, pre):
        mrec = np.r_[0, rec, 1]; mpre = np.r_[0, pre, 0]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())

    out = {}
    for ci, name in enumerate(class_names):
        for thr in thresholds:
            TP, FP, SC, ngt, matched = [], [], [], 0, []
            for tok, per in cache.items():
                ch, sc, ng = per[ci]
                ngt += ng
                if len(sc) == 0:
                    continue
                tp = np.zeros(len(sc)); fp = np.zeros(len(sc))
                if ng and ch.size:
                    arg = ch.argmin(1); best = ch.min(1); taken = set()
                    for i in np.argsort(-sc):
                        if best[i] <= thr and arg[i] not in taken:
                            taken.add(int(arg[i])); tp[i] = 1
                            if thr == chamfer_thr:
                                matched.append(best[i])
                        else:
                            fp[i] = 1
                else:
                    fp[...] = 1
                TP.append(tp); FP.append(fp); SC.append(sc)
            if not SC:
                continue
            tp = np.concatenate(TP); fp = np.concatenate(FP); sc = np.concatenate(SC)
            o = np.argsort(-sc); tpc = np.cumsum(tp[o]); fpc = np.cumsum(fp[o])
            rec = tpc / max(ngt, 1); pre = tpc / np.maximum(tpc + fpc, 1e-12)
            F = 2 * pre * rec / np.maximum(pre + rec, 1e-12)
            k = int(np.argmax(F))
            out[f'{name}_precision_thr_{thr}'] = float(pre[k])
            out[f'{name}_recall_thr_{thr}'] = float(rec[k])
            out[f'{name}_F1_thr_{thr}'] = float(F[k])
            out[f'{name}_APcheck_thr_{thr}'] = voc_ap(rec, pre)
            if thr == chamfer_thr and matched:
                out[f'{name}_chamfer'] = float(np.mean(matched))
                out[f'{name}_chamfer_median'] = float(np.median(matched))
    for key in ('precision', 'recall', 'F1'):
        for thr in thresholds:
            vals = [out[f'{n}_{key}_thr_{thr}'] for n in class_names
                    if f'{n}_{key}_thr_{thr}' in out]
            if vals:
                out[f'm{key}_thr_{thr}'] = float(np.mean(vals))
    ch = [out[f'{n}_chamfer'] for n in class_names if f'{n}_chamfer' in out]
    if ch:
        out['mean_chamfer'] = float(np.mean(ch))
    return out
