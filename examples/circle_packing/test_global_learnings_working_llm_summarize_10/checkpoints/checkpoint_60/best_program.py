# EVOLVE-BLOCK-START
"""Deterministic constructor for 26 circles in the unit square.
Improved explicit pattern with a wider, deterministic search over
horizontal spacing, anisotropic vertical ratio, per-row shifts and
small translations. Uses the exact maximal radii for fixed centers:
r_i = min(dist_to_border, 0.5 * nearest_neighbor_distance).

This version does a coarse grid search and then a simple deterministic
coordinate-descent refinement to squeeze more total radius out of the
same near-hex cluster layout.
"""
import numpy as np
from itertools import product

def construct_packing():
    n = 26
    rows = [5, 5, 6, 5, 5]  # compact near-hex cluster
    vstep_hex = np.sqrt(3.0) / 2.0

    # base integer layout (centered about origin in base units)
    base = []
    for r_idx, cnt in enumerate(rows):
        xs = np.arange(cnt) - (cnt - 1) / 2.0
        x_offset = 0.5 * (r_idx % 2)
        y = r_idx * vstep_hex
        for x in xs:
            base.append([x + x_offset, y])
    base = np.array(base)  # (26,2)

    # helper: compute radii for a given centers array
    def compute_radii(centers, margin=0.0):
        # distance to border (consider margin for safety)
        db = np.minimum.reduce([centers[:, 0] - margin,
                                centers[:, 1] - margin,
                                1.0 - margin - centers[:, 0],
                                1.0 - margin - centers[:, 1]])
        # pairwise center distances
        dif = centers[:, None, :] - centers[None, :, :]
        D = np.sqrt(np.maximum(0.0, (dif ** 2).sum(axis=2)))
        np.fill_diagonal(D, np.inf)
        nearest = D.min(axis=1)
        radii = np.minimum(db, 0.5 * nearest)
        return np.maximum(radii, 0.0)

    # Build centers from parameters: horizontal spacing s_h, vertical ratio r_v,
    # per-row shifts (absolute units), and translation (dx,dy)
    def build_centers(s_h, v_ratio, row_shifts, dx=0.0, dy=0.0, margin=1e-9):
        s_v = v_ratio * vstep_hex * s_h  # absolute vertical step
        coords = base.copy()
        # scale: base horizontal units become s_h, vertical base units become s_v/vstep_hex
        coords[:, 0] = coords[:, 0] * s_h
        coords[:, 1] = coords[:, 1] * (s_v / vstep_hex)
        # apply per-row shifts
        idx = 0
        for r, cnt in enumerate(rows):
            inds = slice(idx, idx + cnt)
            coords[inds, 0] += row_shifts[r]
            idx += cnt
        # center pattern in the unit square (before applying dx,dy) so translations are meaningful
        lo = coords.min(axis=0); hi = coords.max(axis=0)
        trans = 0.5 - 0.5 * (lo + hi)
        coords = coords + trans
        # apply small translation and clip to margin inside square
        coords = coords + np.array([dx, dy])
        coords = np.clip(coords, margin, 1.0 - margin)
        return coords

    # Coarse search ranges (deterministic)
    # horizontal spacing range chosen to allow tight packing but avoid overlaps that force tiny radii
    s_h_vals = np.linspace(0.095, 0.185, 20)      # 20 values
    v_ratio_vals = np.linspace(0.94, 1.06, 7)     # vertical anisotropy multiplier
    shift_fracs = [-0.045, 0.0, 0.045]            # absolute row shifts (will be scaled below)
    # We'll interpret these as fractions of s_h (so shift = frac * s_h)
    # small translation candidates (to nudge cluster into corners/edges)
    trans_grid = [(0.0, 0.0),
                  (-0.002, 0.0), (0.002, 0.0),
                  (0.0, -0.002), (0.0, 0.002),
                  (-0.002, -0.002), (0.002, 0.002)]

    best_sum = -1.0
    best_params = None
    best_centers = None
    best_radii = None

    # Pre-generate all combinations of per-row shift fractions (3^5 = 243)
    all_shift_combos = list(product(shift_fracs, repeat=len(rows)))

    # Coarse grid search
    for s_h in s_h_vals:
        for v_ratio in v_ratio_vals:
            # compute row shifts in absolute units = frac * s_h
            for frac_combo in all_shift_combos:
                row_shifts = np.array(frac_combo) * s_h
                # try a few translations too
                for dx, dy in trans_grid:
                    centers = build_centers(s_h, v_ratio, row_shifts, dx=dx, dy=dy)
                    r = compute_radii(centers, margin=1e-9)
                    ssum = float(r.sum())
                    if ssum > best_sum + 1e-12:
                        best_sum = ssum
                        best_params = (s_h, v_ratio, row_shifts.copy(), dx, dy)
                        best_centers = centers.copy()
                        best_radii = r.copy()

    # Deterministic coordinate-descent refinement around best found coarse params
    if best_params is None:
        # fallback: place a modestly scaled base centered
        s_h = 0.12
        v_ratio = 1.0
        row_shifts = np.zeros(len(rows))
        best_centers = build_centers(s_h, v_ratio, row_shifts)
        best_radii = compute_radii(best_centers)
        best_sum = float(best_radii.sum())
        best_params = (s_h, v_ratio, row_shifts.copy(), 0.0, 0.0)

    # unpack
    cur_s_h, cur_v_ratio, cur_row_shifts, cur_dx, cur_dy = best_params

    # refinement schedule: progressively smaller step sizes
    s_h_steps = [0.012, 0.006, 0.003, 0.0015]
    v_ratio_steps = [0.03, 0.015, 0.007, 0.0035]
    row_shift_steps = [0.03 * cur_s_h, 0.015 * cur_s_h, 0.007 * cur_s_h, 0.003 * cur_s_h]
    trans_steps = [0.004, 0.002, 0.001, 0.0005]

    # deterministic local search: try increasing/decreasing each parameter if improves sum
    for level in range(len(s_h_steps)):
        improved = True
        iter_count = 0
        while improved and iter_count < 6:
            improved = False
            iter_count += 1
            # try s_h adjustments
            for dx_mult in (-1, 1):
                cand_s_h = cur_s_h + dx_mult * s_h_steps[level]
                if cand_s_h <= 0.04 or cand_s_h >= 0.35:
                    continue
                centers = build_centers(cand_s_h, cur_v_ratio, cur_row_shifts, cur_dx, cur_dy)
                r = compute_radii(centers)
                ssum = float(r.sum())
                if ssum > best_sum + 1e-12:
                    best_sum = ssum
                    best_centers = centers.copy()
                    best_radii = r.copy()
                    cur_s_h = cand_s_h
                    improved = True
            # try v_ratio adjustments
            for dv in (-1, 1):
                cand_v = cur_v_ratio + dv * v_ratio_steps[level]
                if cand_v <= 0.7 or cand_v >= 1.4:
                    continue
                centers = build_centers(cur_s_h, cand_v, cur_row_shifts, cur_dx, cur_dy)
                r = compute_radii(centers)
                ssum = float(r.sum())
                if ssum > best_sum + 1e-12:
                    best_sum = ssum
                    best_centers = centers.copy()
                    best_radii = r.copy()
                    cur_v_ratio = cand_v
                    improved = True
            # try per-row shift adjustments (each row independently)
            for r_idx in range(len(rows)):
                for dir_mult in (-1, 1):
                    cand_row_shifts = cur_row_shifts.copy()
                    step = row_shift_steps[level]
                    cand_row_shifts[r_idx] += dir_mult * step
                    centers = build_centers(cur_s_h, cur_v_ratio, cand_row_shifts, cur_dx, cur_dy)
                    r = compute_radii(centers)
                    ssum = float(r.sum())
                    if ssum > best_sum + 1e-12:
                        best_sum = ssum
                        best_centers = centers.copy()
                        best_radii = r.copy()
                        cur_row_shifts = cand_row_shifts
                        improved = True
            # try translations
            for tx_mult, ty_mult in product((-1, 0, 1), repeat=2):
                if tx_mult == 0 and ty_mult == 0:
                    continue
                cand_dx = cur_dx + tx_mult * trans_steps[level]
                cand_dy = cur_dy + ty_mult * trans_steps[level]
                centers = build_centers(cur_s_h, cur_v_ratio, cur_row_shifts, cand_dx, cand_dy)
                r = compute_radii(centers)
                ssum = float(r.sum())
                if ssum > best_sum + 1e-12:
                    best_sum = ssum
                    best_centers = centers.copy()
                    best_radii = r.copy()
                    cur_dx, cur_dy = cand_dx, cand_dy
                    improved = True

    # Final tiny nudges deterministically (very small translations)
    for dx_try in (-5e-4, 0.0, 5e-4):
        for dy_try in (-5e-4, 0.0, 5e-4):
            centers = build_centers(cur_s_h, cur_v_ratio, cur_row_shifts, cur_dx + dx_try, cur_dy + dy_try)
            r = compute_radii(centers)
            ssum = float(r.sum())
            if ssum > best_sum + 1e-12:
                best_sum = ssum
                best_centers = centers.copy()
                best_radii = r.copy()
                cur_dx += dx_try; cur_dy += dy_try

    # Safety clamp if any negative or zero radii
    if best_radii is None:
        best_centers = build_centers(0.12, 1.0, np.zeros(len(rows)))
        best_radii = compute_radii(best_centers)

    return np.array(best_centers), np.array(best_radii), float(best_radii.sum())

# EVOLVE-BLOCK-END

def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")