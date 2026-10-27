"""Polymer–colloid adsorption, bridging and floc charge (Modules 1–2).

Definitions
-----------
adsorbed monomer : centre within ``r_ads`` of a colloid centre (assigned to the nearest colloid)
attached chain   : a chain with >= ``min_contacts`` adsorbed monomers on a given colloid
                   (filters out transient single-bead touches)
bridging chain   : a chain attached to >= 2 distinct colloids
train/loop/tail  : Scheutjens–Fleer decomposition of an adsorbed chain into runs of
                   adsorbed monomers (trains), non-adsorbed runs between trains (loops)
                   and non-adsorbed chain ends (tails)
floc             : connected component of the graph whose edges are colloid contacts
                   (surface gap < gap) *or* shared bridging chains
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .clusters import _wrap, contact_pairs, labels_from_pairs

COLLOID, NEUTRAL, CATIONIC = 1, 2, 3


def adsorption_map(colloid_pos, monomer_pos, box_len, r_ads: float, box_lo=None) -> np.ndarray:
    """Index of the colloid each monomer is adsorbed on (-1 if free)."""
    box_len = np.broadcast_to(np.asarray(box_len, dtype=float), (3,))
    lo = np.zeros(3) if box_lo is None else np.asarray(box_lo, dtype=float)
    owner = np.full(len(monomer_pos), -1, dtype=int)
    if len(monomer_pos) == 0 or len(colloid_pos) == 0:
        return owner
    m = _wrap(monomer_pos, lo, box_len)
    c = _wrap(colloid_pos, lo, box_len)
    best = np.full(len(m), np.inf)
    tree = cKDTree(m, boxsize=box_len)
    for k, hits in enumerate(tree.query_ball_point(c, r_ads)):
        if not hits:
            continue
        hits = np.asarray(hits)
        d = m[hits] - c[k]
        d -= box_len * np.round(d / box_len)
        r = np.linalg.norm(d, axis=1)
        closer = r < best[hits]
        owner[hits[closer]] = k
        best[hits[closer]] = r[closer]
    return owner


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """Run-length encoding: (start, stop, value) for consecutive equal entries."""
    if len(mask) == 0:
        return []
    change = np.flatnonzero(np.diff(mask.astype(int))) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(mask)]])
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(starts, stops)]


def train_loop_tail(adsorbed: np.ndarray) -> dict[str, int]:
    """Number of monomers in trains, loops and tails for one chain."""
    out = {"train": 0, "loop": 0, "tail": 0, "n_trains": 0, "n_loops": 0}
    if not adsorbed.any():
        return out
    runs = _runs(adsorbed)
    for i, (a, b, ads) in enumerate(runs):
        n = b - a
        if ads:
            out["train"] += n
            out["n_trains"] += 1
        elif i == 0 or i == len(runs) - 1:
            out["tail"] += n
        else:
            out["loop"] += n
            out["n_loops"] += 1
    return out


def analyze_frame(frame, meta: dict, r_ads: float | None = None, gap: float = 2.0,
                  min_contacts: int = 2) -> dict:
    """Adsorption, bridging and floc statistics for one dump frame of Module 1 or 2."""
    dc = float(meta["params"]["colloid_diameter"])
    z = float(meta["params"]["colloid_charge"])
    if r_ads is None:
        r_ads = 0.5 * (dc + 1.0) + 1.0   # one sigma beyond the colloid–monomer contact
    order = np.argsort(frame["id"])
    types = frame["type"][order]
    mol = frame["mol"][order]
    pos = frame.positions("x")[order]
    L, lo = frame.box_lengths, frame.box_lo

    is_c = types == COLLOID
    cpos, mpos = pos[is_c], pos[~is_c]
    mtype, mmol = types[~is_c], mol[~is_c]
    n_c = len(cpos)
    owner = adsorption_map(cpos, mpos, L, r_ads, lo)

    chains = np.unique(mmol)
    chain_colloids, chain_charge = [], []
    seg = {"train": 0, "loop": 0, "tail": 0, "n_trains": 0, "n_loops": 0}
    n_ads_chains = 0
    for ch in chains:
        sel = mmol == ch
        own = owner[sel]
        counts = np.bincount(own[own >= 0], minlength=n_c)
        chain_colloids.append(set(np.flatnonzero(counts >= min_contacts).tolist()))
        chain_charge.append(float(np.sum(mtype[sel] == CATIONIC)))
        if chain_colloids[-1]:
            n_ads_chains += 1
            for k, v in train_loop_tail(own >= 0).items():
                seg[k] += v
    n_bridging = sum(len(s) >= 2 for s in chain_colloids)

    # floc graph: geometric contacts + bridges
    edges = contact_pairs(cpos, L, dc + gap, lo).tolist()
    for s in chain_colloids:
        edges += [list(p) for p in combinations(sorted(s), 2)]
    labels = labels_from_pairs(n_c, np.array(edges, dtype=int).reshape(-1, 2))
    contact_labels = labels_from_pairs(n_c, contact_pairs(cpos, L, dc + gap, lo))

    # net charge of each floc: colloids (-Z) + every chain adsorbed on it (whole chain)
    q_floc = -z * np.bincount(labels, minlength=labels.max() + 1 if n_c else 0).astype(float)
    for s, q in zip(chain_colloids, chain_charge):
        if s:
            q_floc[labels[next(iter(s))]] += q
    q_per_colloid = q_floc[labels] / np.bincount(labels)[labels] if n_c else np.zeros(0)

    sizes = np.bincount(labels) if n_c else np.zeros(0)
    csizes = np.bincount(contact_labels) if n_c else np.zeros(0)
    n_mono_ads = int((owner >= 0).sum())
    n_mono_in_ads_chains = seg["train"] + seg["loop"] + seg["tail"]
    return {
        "timestep": frame.timestep,
        "n_colloids": n_c,
        "n_chains": int(len(chains)),
        "n_flocs": int(len(sizes)),
        "floc_size_weight": float((sizes**2).sum() / sizes.sum()) if n_c else 0.0,
        "floc_size_max": int(sizes.max()) if n_c else 0,
        "n_contact_clusters": int(len(csizes)),
        "contact_size_weight": float((csizes**2).sum() / csizes.sum()) if n_c else 0.0,
        "frac_chains_adsorbed": n_ads_chains / max(1, len(chains)),
        "frac_chains_bridging": n_bridging / max(1, len(chains)),
        "bridges_per_colloid": 2.0 * sum(len(s) * (len(s) - 1) / 2 for s in chain_colloids) / max(1, n_c),
        "frac_monomers_adsorbed": n_mono_ads / max(1, len(mpos)),
        "frac_train": seg["train"] / max(1, n_mono_in_ads_chains),
        "frac_loop": seg["loop"] / max(1, n_mono_in_ads_chains),
        "frac_tail": seg["tail"] / max(1, n_mono_in_ads_chains),
        "mean_loop_length": seg["loop"] / max(1, seg["n_loops"]),
        "net_charge_per_colloid": float(q_per_colloid.mean()) if n_c else 0.0,
        "net_charge_per_colloid_std": float(q_per_colloid.std()) if n_c else 0.0,
    }
