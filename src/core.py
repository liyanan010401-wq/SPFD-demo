"""Synthetic CPU mechanism example, NOT the DEIM-D-FINE research implementation."""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

EPS = 1e-8

def seed_all(seed, threads=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)

def preprocess(images, valid, train):
    """Per-window valid-pixel min/max, then source-train-only z-score; invalid=0."""
    scaled = torch.zeros_like(images)
    for i in range(len(images)):
        values = images[i][valid[i]]
        if not values.numel():
            raise ValueError("A synthetic window has no valid pixels")
        span = (values.max() - values.min()).clamp_min(EPS)
        scaled[i] = torch.where(valid[i], (images[i] - values.min()) / span, 0.)
    training_values = scaled[train][valid[train]]
    mean = training_values.mean()
    std = training_values.std(unbiased=False).clamp_min(EPS)
    result = torch.where(valid, (scaled - mean) / std, 0.)
    return result, {"mean": float(mean), "std": float(std),
                    "fit_split": "train", "statistic_pixel_count": training_values.numel(),
                    "order": "window minmax -> train zscore -> FFT",
                    "invalid_value": 0.0}

def xyxy(b):
    return torch.cat((b[..., :2] - b[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2), -1)

def giou(a, b):
    a, b = xyxy(a), xyxy(b)
    lo, hi = torch.maximum(a[:, :2], b[:, :2]), torch.minimum(a[:, 2:], b[:, 2:])
    inter = (hi - lo).clamp_min(0).prod(-1)
    union = (a[:, 2:] - a[:, :2]).prod(-1) + (b[:, 2:] - b[:, :2]).prod(-1) - inter
    enc = (torch.maximum(a[:, 2:], b[:, 2:]) - torch.minimum(a[:, :2], b[:, :2])).prod(-1)
    return inter / union.clamp_min(EPS) - (enc - union) / enc.clamp_min(EPS)

class TinyDetector(nn.Module):
    """Three queries; one 16-channel stride-2 demo feature, NOT formal stride-16."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 12, 3, padding=1)
        self.c2 = nn.Conv2d(12, 16, 3, padding=1)
        self.head = nn.Linear(16 * 4 * 4, 3 * 5)
    def forward(self, x):
        f1 = F.gelu(self.c1(x))
        feature = F.gelu(self.c2(F.avg_pool2d(f1, 2)))
        out = self.head(F.adaptive_avg_pool2d(feature, 4).flatten(1)).view(-1, 3, 5)
        return out[:, :, 0], out[:, :, 1:].sigmoid(), feature

def assignment(logits, boxes, target):
    if not len(target):
        return np.empty(0, int)
    if len(target) > len(boxes):
        raise ValueError("TinyDetector permits at most three GT instances")
    cost = torch.cdist(boxes, target, p=1) - logits.sigmoid()[:, None]
    q, j = linear_sum_assignment(cost.detach().cpu().numpy())
    return q[np.argsort(j)]

def supervised(logits, boxes, target):
    """BCE + 3*mean4 L1 + GIoU surrogate; not MAL or native DEIM criterion."""
    ix = assignment(logits, boxes, target)
    labels = torch.zeros_like(logits)
    labels[ix] = 1
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    if len(ix):
        loss = loss + 3 * F.l1_loss(boxes[ix], target) + (1 - giou(boxes[ix], target)).mean()
    return loss, ix

def consistency(l0, b0, l1, b1, target):
    i0, i1 = assignment(l0, b0, target), assignment(l1, b1, target)
    if not len(target):
        return (l0.sum() + l1.sum() + b0.sum() + b1.sum()) * 0
    p, q = l0[i0].sigmoid(), l1[i1].sigmoid()
    p, q = torch.stack((p, 1-p), -1), torch.stack((q, 1-q), -1)
    p, q = p.clamp_min(EPS), q.clamp_min(EPS)
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    m = (p + q) / 2
    js = .5 * (p * (p/m).log() + q * (q/m).log()).sum(-1).mean()
    # Match the confirmed sum4-then-mean query term, beta_b=1.
    return js + (b0[i0] - b1[i1]).abs().sum(-1).mean()

def spectral_masks(x, k=8, radial_bins=8, angular_bins=8, seed=17, max_iter=100):
    """Conjugate-paired cells and 1D K-means++ on mean log1p amplitudes."""
    h, w = x.shape
    amp = torch.fft.fft2(x).abs()
    y, z = np.mgrid[:h, :w]
    flat, conjugate = y*w + z, ((-y) % h)*w + ((-z) % w)
    canonical = np.minimum(flat, conjugate)
    fy = np.fft.fftfreq(h)[canonical // w]
    fx = np.fft.fftfreq(w)[canonical % w]
    radius = np.hypot(fy, fx) / np.sqrt(.5)
    radial = np.minimum((radius * radial_bins).astype(int), radial_bins - 1)
    angular = np.minimum(((np.arctan2(fy, fx)+np.pi)/(2*np.pi)*angular_bins).astype(int), angular_bins-1)
    cells = radial * angular_bins + angular
    keys = np.unique(cells)
    values = np.array([torch.log1p(amp)[torch.from_numpy(cells == c)].mean().item() for c in keys])
    k = min(k, len(keys))
    rng = np.random.default_rng(seed)
    indices = [int(rng.integers(len(values)))]
    for _ in range(1, k):
        residual = ((values[:, None] - values[indices][None, :])**2).min(axis=1)
        residual[indices] = 0
        if residual.sum() > 0:
            indices.append(int(rng.choice(len(values), p=residual/residual.sum())))
        else:
            indices.append(next(i for i in range(len(values)) if i not in indices))
    centers = values[indices].copy()
    for _ in range(max_iter):
        distance = abs(values[:, None] - centers[None, :])
        labels = distance.argmin(axis=1)
        new = centers.copy()
        for j in range(k):
            if np.any(labels == j):
                new[j] = values[labels == j].mean()
            else:
                residual = ((values[:, None] - new[None, :])**2).min(axis=1)
                new[j] = values[int(residual.argmax())]
        if np.allclose(new, centers, rtol=0, atol=1e-12):
            centers = new
            break
        centers = new
    labels = abs(values[:, None] - centers[None, :]).argmin(axis=1)
    return [torch.from_numpy(np.isin(cells, keys[labels == j])).to(x)
            for j in range(k) if np.any(labels == j)]

def protection(boxes, h, w, rho=.25):
    if rho <= 0:
        raise ValueError("rho must be positive")
    y, x = torch.meshgrid((torch.arange(h)+.5)/h, (torch.arange(w)+.5)/w, indexing="ij")
    out = torch.zeros(h, w)
    for cx, cy, bw, bh in boxes.detach():
        if bw <= 0 or bh <= 0:
            continue
        distance = torch.maximum(2*abs(x-cx)/bw, 2*abs(y-cy)/bh)
        profile = torch.where(distance <= 1, 1.,
                    torch.where(distance < 1+rho, .5*(1+torch.cos(torch.pi*(distance-1)/rho)), 0.))
        out = torch.maximum(out, profile)
    return out

def mix_restore(x, donor, mask, boxes, valid, xi=.5, rho=.25, protected=True):
    spectrum = torch.fft.fft2(x)
    donor_amp = torch.fft.fft2(donor).abs()
    amplitude = mask*spectrum.abs() + (1-mask)*((1-xi)*spectrum.abs()+xi*donor_amp)
    # Confirmed order: amplitude clip BEFORE iFFT. No reconstructed-pixel clip.
    amplitude = amplitude.clamp(0, 2)
    raw = torch.fft.ifft2(amplitude * torch.exp(1j*torch.angle(spectrum)))
    p = protection(boxes, *x.shape, rho=rho) * valid if protected else torch.zeros_like(x)
    result = torch.where(valid, p*x + (1-p)*raw.real, x)
    return result, p, float(raw.imag.abs().max())

def legal_donors(i, train, window, frequency):
    return [j for j in train if window[j] != window[i] and frequency[j] == frequency[i]]

def roi_vectors(feature, rois, grid_size=7):
    """Single-layer 7x7 center sampling; deliberately NOT native RoIAlign ratio=2."""
    if not len(rois):
        return feature.new_empty((0, feature.shape[1]))
    grids = []
    for r in rois.detach():
        x1, y1, x2, y2 = r
        ys = y1 + (torch.arange(grid_size)+.5)*(y2-y1)/grid_size
        xs = x1 + (torch.arange(grid_size)+.5)*(x2-x1)/grid_size
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grids.append(torch.stack((xx*2-1, yy*2-1), -1))
    grid = torch.stack(grids)
    return F.grid_sample(feature.expand(len(rois), -1, -1, -1), grid,
                         align_corners=False).mean((2,3))

class RegionalHead(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.proj = nn.Linear(16, dimension)
        self.directions = nn.Parameter(torch.randn(2, dimension))
        self.raw_k = nn.Parameter(torch.tensor(.5))
    def forward(self, vectors):
        z = self.proj(vectors)
        norm = z.norm(dim=-1, keepdim=True)
        direction_norm = self.directions.norm(dim=-1, keepdim=True)
        valid = (norm[:, 0] >= EPS) & (direction_norm >= EPS).all()
        unit_z = z / norm.clamp_min(EPS)
        unit_d = self.directions / direction_norm.clamp_min(EPS)
        logits = (unit_z @ unit_d.T) * F.softplus(self.raw_k)
        return logits.softmax(-1), valid

def posterior_loss(q0, q1, labels, valid):
    if not valid.any():
        return (q0.sum() + q1.sum()) * 0
    a, b = q0.clamp_min(EPS), q1.clamp_min(EPS)
    ce = -.5 * (a[range(len(labels)), labels].log() + b[range(len(labels)), labels].log())
    teacher = a.detach()  # Only clean KL branch stops gradients; CE remains live.
    kl = (teacher*(teacher.log()-b.log())).sum(-1)
    per_region = ce + kl
    return torch.stack([per_region[valid & (labels == c)].mean()
                        for c in labels[valid].unique()]).mean()

def regions(target, valid, rho=.25):
    """At most two GT RoIs + one fixed valid background candidate: demo reduction."""
    rs = list(xyxy(target).clamp(0, 1))
    labels = [1]*len(rs)
    h, w = valid.shape
    for left in [.05, .45, .85]:
        r = torch.tensor([left, .8, min(left+.1, 1), .9])
        lo = torch.floor(r[:2]*torch.tensor([w,h])).long().clamp_min(0)
        hi = torch.ceil(r[2:]*torch.tensor([w,h])).long()
        if not valid[lo[1]:hi[1], lo[0]:hi[0]].all():
            continue
        expanded = xyxy(target * target.new_tensor([1,1,1+rho,1+rho]))
        if len(expanded) and (((torch.minimum(r[2:], expanded[:,2:]) -
                               torch.maximum(r[:2], expanded[:,:2])) > 0).all(-1)).any():
            continue
        rs.append(r); labels.append(0)
        break
    return (torch.stack(rs) if rs else torch.empty(0,4)), torch.tensor(labels, dtype=torch.long)

def run(path, out, cfg):
    if cfg.get("device") != "cpu":
        raise ValueError("This delivery is a CPU-only mechanism example")
    seed_all(cfg["seed"], cfg.get("torch_threads", 1))
    data = np.load(path, allow_pickle=False)
    if not bool(data["synthetic_data"]):
        raise ValueError("This runner accepts the bundled synthetic example only")
    valid = torch.tensor(data["valid"])
    train = np.where(data["split"] == "train")[0]
    dev = np.where(data["split"] == "dev")[0]
    test = np.where(data["split"] == "test")[0]
    groups = [set(data["group_id"][ix]) for ix in [train, dev, test]]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i+1,3))
    x, normalization = preprocess(torch.tensor(data["images"]), valid, train)
    target = lambda i: torch.tensor(data["boxes"][i, :int(data["counts"][i])])
    settings = cfg["mechanism"]
    model = TinyDetector()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    warm = []
    for step in range(cfg["warmup_steps"]):
        i = int(train[step % len(train)])
        logits, boxes, _ = model(x[i:i+1])
        loss, _ = supervised(logits[0], boxes[0], target(i))
        opt.zero_grad(); loss.backward(); opt.step()
        warm.append(float(loss.detach()))
    selector = copy.deepcopy(model).eval()
    for param in selector.parameters():
        param.requires_grad_(False)
    frozen = {k:v.detach().clone() for k,v in selector.state_dict().items()}
    cache, cache_records, stored_masks = {}, [], []
    with torch.no_grad():
        for i in train:
            masks = spectral_masks(x[i,0], k=settings["spectral_clusters"],
                                   radial_bins=8, angular_bins=8, seed=17, max_iter=100)
            scores = []
            for mask in masks:
                retained = torch.fft.ifft2(torch.fft.fft2(x[i,0])*mask).real
                retained = torch.where(valid[i,0], retained, 0.)
                logits, boxes, _ = selector(retained[None,None])
                scores.append(float(supervised(logits[0], boxes[0], target(i))[0]))
            order = np.argsort(-np.array(scores), kind="stable")[:settings["cached_top_masks"]]
            cache[int(i)] = [masks[k] for k in order]
            cache_records.append({"window_id":int(data["window_id"][i]),
                                  "scores":scores, "selected_cluster_indices":order.tolist()})
            stored_masks.append(torch.stack(cache[int(i)]).numpy())
    head = RegionalHead(settings["regional_dimension"])
    opt = torch.optim.AdamW(list(model.parameters())+list(head.parameters()), lr=cfg["learning_rate"])
    history, trace = [], []
    rng = np.random.default_rng(cfg["seed"])
    max_imaginary, region_count = 0., 0
    for step in range(cfg["train_steps"]):
        i = int(train[step % len(train)])
        donors = legal_donors(i, train, data["window_id"], data["frequency"])
        if not donors:
            raise ValueError("No legal same-frequency source-training donor")
        donor = int(rng.choice(donors))
        slot = int(rng.integers(len(cache[i])))
        xi = float(rng.uniform(0, settings["xi_max"]))
        gt = target(i)
        augmented, p, imag = mix_restore(x[i,0], x[donor,0], cache[i][slot], gt, valid[i,0],
                                         xi=xi, rho=settings["protection_rho"])
        max_imaginary = max(max_imaginary, imag)
        pair = torch.stack((x[i,0], augmented))[:,None]
        logits, boxes, feature = model(pair)
        det = (supervised(logits[0],boxes[0],gt)[0]+supervised(logits[1],boxes[1],gt)[0])/2
        query = consistency(logits[0],boxes[0],logits[1],boxes[1],gt)
        rois, labels = regions(gt, valid[i,0], settings["protection_rho"])
        qa, va = head(roi_vectors(feature[:1],rois,settings["roi_grid"]))
        qb, vb = head(roi_vectors(feature[1:],rois,settings["roi_grid"]))
        regional = posterior_loss(qa,qb,labels,va&vb)
        region_count += len(labels)
        loss = det + settings["query_weight"]*query + settings["regional_weight"]*regional
        opt.zero_grad(); loss.backward(); opt.step()
        history.append({"total":float(loss.detach()),"query":float(query.detach()),
                        "regional":float(regional.detach())})
        trace.append({"step":step,"source_window_id":int(data["window_id"][i]),
                      "donor_window_id":int(data["window_id"][donor]),"cache_slot":slot,"xi":xi,
                      "protection_mass":float(p[valid[i,0]].mean()),
                      "valid_pixel_RMS":float((augmented[valid[i,0]]-x[i,0][valid[i,0]]).square().mean().sqrt()),
                      "synthetic_only":True})
    last_valid = valid[i,0].clone()
    model.eval()
    with torch.no_grad():
        logits, boxes, _ = model(x[test])
        error = []
        for n, i in enumerate(test):
            matched = assignment(logits[n],boxes[n],target(i))
            if len(matched):
                error.append(float(abs(boxes[n,matched]-target(i)).mean()))
    np.savez(out/"synthetic_predictions.npz", scores=logits.sigmoid().numpy(),
             boxes_cxcywh=boxes.numpy(), sample_ids=test,
             synthetic_data=True, publication_evidence=False)
    np.savez(out/"synthetic_augmentation.npz", original=pair[0,0].numpy(),
             augmented=augmented.numpy(), protection=p.numpy(), valid=last_valid.numpy(),
             synthetic_data=True, publication_evidence=False)
    np.savez(out/"synthetic_cached_masks.npz", masks=np.stack(stored_masks),
             source_sample_ids=train, synthetic_data=True, publication_evidence=False)
    with (out/"synthetic_model.pt").open("wb") as stream:
        torch.save({"detector":model.state_dict(),"normalization":normalization,
                    "synthetic_only":True,"publication_evidence":False},stream)
    (out/"synthetic_trace.json").write_text(json.dumps(
        {"synthetic_only":True,"publication_evidence":False,"normalization":normalization,
         "selector_cache":cache_records,"training_trace":trace},indent=2),encoding="utf-8")
    return {"warmup_losses":warm,"joint_losses":history,
            "synthetic_test_matched_box_mae_normalized":float(np.mean(error)),
            "test_count":len(test),"selector_unchanged":all(torch.equal(v,selector.state_dict()[k]) for k,v in frozen.items()),
            "selector_no_grad":all(p.grad is None for p in selector.parameters()),
            "maximum_ifft_imaginary":max_imaginary,"regions_trained":region_count,
            "inference_uses_selector":False,"inference_uses_regional_head":False,
            "normalization":normalization,"learned_kappa":float(F.softplus(head.raw_k).detach()),
            "limitations":["TinyDetector is not DEIM-D-FINE; BCE/L1/GIoU surrogate is not native MAL.",
                           "Single 16-channel stride-2 feature and grid_sample approximate the regional interface; not formal RoIAlign.",
                           "Identity composition, no four-window Dense O2O mosaic or full-window stitching/NMS.",
                           "Synthetic textures are not sonar or physical-frequency transfer; no publication AP or CI is computed."]}
