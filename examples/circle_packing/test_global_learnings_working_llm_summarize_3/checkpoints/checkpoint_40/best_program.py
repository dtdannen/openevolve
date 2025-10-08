# EVOLVE-BLOCK-START
"""Improved deterministic constructor for n=26 circle packing in the unit square.

Strategy:
- Build hexagonal (offset) lattice of candidate centers for a spacing s found by binary search.
- Try a small set of sub-grid shifts to mitigate alignment/edge effects.
- For each candidate center set (take the 26 centers closest to the square center),
  compute maximal feasible radii by a fast iterative relaxation:
    r_i <- min(border_limit_i, min_j (dist_ij - r_j))
  which enforces r_i + r_j <= dist_ij and r_i <= border limits.
- Pick the candidate with largest sum of radii.
This vectorized projection typically yields denser radii than the conservative
0.5*nearest-neighbor rule.
"""
import numpy as np

def construct_packing():
    n_target = 26

    def make_hex_centers(s, shift=(0.0, 0.0)):
        # margin ensures centers are at least s/2 from border when unshifted;
        # the shift can be up to ~s/2 to explore offsets.
        margin = s * 0.5
        pts = []
        y_step = s * np.sqrt(3.0) / 2.0
        # start from margin plus shift
        y = margin + shift[1]
        row = 0
        while y <= 1.0 - margin + 1e-12:
            x_off = (row % 2) * (s / 2.0)
            x = margin + x_off + shift[0]
            while x <= 1.0 - margin + 1e-12:
                # clamp tiny bit inside to avoid exact border equality
                pts.append((min(max(x,1e-12), 1.0-1e-12), min(max(y,1e-12), 1.0-1e-12)))
                x += s
            row += 1
            y += y_step
        return np.array(pts, dtype=float)

    # Binary search largest spacing s that yields at least n_target centers
    lo, hi = 0.01, 0.6
    best_s = lo
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if make_hex_centers(mid).shape[0] >= n_target:
            best_s = mid
            lo = mid
        else:
            hi = mid

    # Try small shifts around (0,0) to reduce edge alignment penalties.
    # Use a deterministic list of shifts scaled to s/4.
    shifts = [(0.0, 0.0), (0.25, 0.0), (-0.25, 0.0), (0.0,0.25), (0.0,-0.25)]
    shifts = [(sx * best_s, sy * best_s) for (sx, sy) in shifts]

    best_sum = -1.0
    best_centers = None
    best_radii = None

    for sh in shifts:
        cand = make_hex_centers(best_s, shift=sh)
        if cand.shape[0] < n_target:
            continue
        # pick the n_target centers closest to square center to mitigate edge effects
        center_sq = np.array([0.5, 0.5])
        d = np.linalg.norm(cand - center_sq, axis=1)
        idx = np.argsort(d)[:n_target]
        centers = cand[idx].copy()
        # deterministic ordering
        centers = centers[np.lexsort((centers[:,1], centers[:,0]))]

        radii = _relax_radii(centers)
        ssum = float(np.sum(radii))
        if ssum > best_sum + 1e-12:
            best_sum = ssum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Fallback: if none produced, create the centered hex without shift
    if best_centers is None:
        centers = make_hex_centers(best_s)[:n_target]
        centers = centers[np.lexsort((centers[:,1], centers[:,0]))]
        best_radii = _relax_radii(centers)
        best_centers = centers
        best_sum = float(np.sum(best_radii))

    return best_centers, best_radii, float(best_sum)


def _relax_radii(centers, tol=1e-8, max_iter=5000):
    """Vectorized fixed-point relaxation:
    r_i <- min(border_limit_i, min_j (dist_ij - r_j))
    Ensures r_i + r_j <= dist_ij and r_i <= border_limit.
    """
    n = centers.shape[0]
    # distance to borders
    border = np.minimum.reduce([centers[:,0], centers[:,1], 1.0 - centers[:,0], 1.0 - centers[:,1]])
    border = np.maximum(border, 0.0)
    # pairwise distances
    dif = centers[:, None, :] - centers[None, :, :]
    D = np.sqrt(np.maximum(0.0, np.sum(dif * dif, axis=2)))
    # avoid self in minima
    np.fill_diagonal(D, np.inf)

    # initialize with border limits (optimistic)
    r = border.copy()
    for _ in range(max_iter):
        r_old = r
        # allowed for i given current r_j is min_j (D_ij - r_j)
        allowed = D - r[None, :]   # shape (n,n)
        # take min across j, then clamp
        new_r = np.minimum(border, np.maximum(0.0, np.min(allowed, axis=1)))
        # If any distances are inf (isolated), min yields inf -> clamp to border already
        change = np.max(np.abs(new_r - r))
        r = new_r
        if change < tol:
            break
    # final safety: ensure pairwise constraints (tiny numerical corrections)
    # If any pair still violates, apply a single greedy correction shrinking larger circle.
    I, J = np.triu_indices(n, k=1)
    dij = D[I, J]
    rij = r[I] + r[J]
    mask = rij > dij + 1e-12
    if np.any(mask):
        for i, j, d in zip(I[mask], J[mask], dij[mask]):
            if r[i] + r[j] <= d + 1e-12:
                continue
            if r[i] >= r[j]:
                r[i] = max(0.0, d - r[j])
            else:
                r[j] = max(0.0, d - r[i])
        # enforce border again
        r = np.minimum(r, border)
    return r


# EVOLVE-BLOCK-END


# Runner and visualizer unchanged (interface preserved)
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
    for c,r in zip(centers, radii):
        ax.add_patch(Circle(c, r, alpha=0.6, edgecolor='k'))
    plt.title(f"n={len(centers)}, sum={sum(radii):.6f}")
    plt.show()

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(centers, radii)