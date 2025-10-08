# EVOLVE-BLOCK-START
"""Deterministic constructor for n=26 circles in the unit square.

Improvements over prior versions:
- Small deterministic grid-search over horizontal/vertical spacing scales to
  pick the best-shaped hex-like block for 26 circles.
- A more aggressive growth-then-robust-shrink radii solver that fully expands
  radii toward slack then repeatedly resolves overlaps (stronger convergence
  to a larger feasible sum of radii).
Returns centers (26,2), radii (26,), sum_radii.
"""
import numpy as np

def construct_packing():
    n = 26
    rows = [6, 5, 6, 5, 4]  # total 26
    R = len(rows)
    max_k = max(rows)
    margin = 1e-6
    # base horizontal spacing (widest row)
    if max_k > 1:
        s0 = (1.0 - 2 * margin) / (max_k - 1)
    else:
        s0 = 0.5

    # deterministic candidate scales (tries a few compressions to balance edge effects)
    s_scales = np.array([1.00, 0.98, 0.96, 0.94, 0.92])
    y_scales = np.array([1.00, 0.98, 0.96])

    best_sum = -1.0
    best_centers = None
    best_radii = None

    for ss in s_scales:
        for ys in y_scales:
            s = s0 * ss
            h = s * np.sqrt(3.0) / 2.0 * ys
            # if vertical span too large, rescale to fit exactly
            span_v = (R - 1) * h
            if span_v > 1.0 - 2 * margin:
                h = (1.0 - 2 * margin) / (R - 1)
                s = h * 2.0 / np.sqrt(3.0)
            # build centered rows
            y0 = 0.5 - ((R - 1) * h) / 2.0
            centers = []
            for i, k in enumerate(rows):
                y = y0 + i * h
                left = 0.5 - ((k - 1) * s) / 2.0
                xs = left + np.arange(k) * s
                for x in xs:
                    centers.append([float(x), float(y)])
            centers = np.array(centers[:n], dtype=float)
            # minor safeguard: clip into unit square (numerical)
            centers = np.clip(centers, margin, 1.0 - margin)

            radii = compute_max_radii_aggressive(centers)
            sumr = float(np.sum(radii))
            if sumr > best_sum:
                best_sum = sumr
                best_centers = centers
                best_radii = radii

    return best_centers, best_radii, float(np.sum(best_radii))


def compute_max_radii_aggressive(centers, max_outer=300, max_inner=60, tol=1e-10):
    """
    Aggressive grow-then-shrink solver:
    - start with conservative radii (min border, 0.5*nearest)
    - repeatedly:
        * compute allowed = min(border, min_j (D_ij - r_j)) and set radii = allowed
        * run an inner loop that repeatedly resolves overlaps by proportional scaling
          (using the most restrictive scale per circle) until no overlaps remain
    This often reaches a larger feasible sum than tiny incremental growth steps.
    """
    n = centers.shape[0]
    if n == 0:
        return np.array([])

    # border limits
    border = np.minimum.reduce([centers[:, 0], centers[:, 1],
                                1.0 - centers[:, 0], 1.0 - centers[:, 1]])
    # pairwise distances
    dif = centers[:, None, :] - centers[None, :, :]
    D = np.hypot(dif[..., 0], dif[..., 1])
    np.fill_diagonal(D, np.inf)

    # initial safe radii
    min_pair = np.min(D, axis=1)
    radii = np.minimum(border, 0.5 * min_pair)
    radii = np.maximum(radii, 0.0)

    for outer in range(max_outer):
        old = radii.copy()

        # compute allowed growth (full assignment)
        slack = D - radii[None, :]   # allowed remaining toward each other
        allowed = np.minimum(border, np.min(slack, axis=1))
        allowed = np.maximum(allowed, 0.0)
        radii = allowed.copy()

        # inner robust shrink: resolve overlaps thoroughly
        for inner in range(max_inner):
            ri = radii[:, None]
            rj = radii[None, :]
            sumr = ri + rj
            viol = sumr > D + 1e-14
            if not np.any(viol):
                break
            # per-pair scale factors (<=1)
            factors = np.ones_like(sumr)
            factors[viol] = D[viol] / sumr[viol]
            # per-circle minimal scale
            min_scale = factors.min(axis=1)
            # clamp to avoid underflow
            min_scale = np.maximum(min_scale, 1e-12)
            radii *= min_scale
            # ensure wall limits
            radii = np.minimum(radii, border)

        # small relaxation to avoid infinite oscillation:
        radii = np.minimum(radii, border)

        if np.max(np.abs(radii - old)) < tol:
            break

    radii = np.maximum(radii, 0.0)
    return radii

# EVOLVE-BLOCK-END

def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")