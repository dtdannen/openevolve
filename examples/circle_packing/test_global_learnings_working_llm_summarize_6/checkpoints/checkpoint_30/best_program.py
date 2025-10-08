# EVOLVE-BLOCK-START
"""Improved explicit constructor for n=26 circle packing in the unit square.

This version:
- Enumerates several hex-like row patterns and small variations (row offsets,
  slight vertical compression/expansion, small translations).
- For each candidate it places deterministic centers and computes maximal
  feasible radii using a robust Gauss-Seidel projection followed by
  conservative growth cycles and proportional fixes.
- Returns the best layout found (max sum of radii).
"""
import numpy as np

def construct_packing():
    n = 26
    # candidate row patterns summing to 26, chosen to break perfect symmetry
    patterns = [
        [6,5,5,5,5],
        [5,6,5,5,5],
        [6,5,6,5,4],
        [5,6,6,5,4],
        [4,6,6,5,5],
    ]

    # try a grid of horizontal spacings and small vertical scalings
    s_vals = np.linspace(0.12, 0.20, 17)
    y_scales = [0.94, 0.98, 1.0, 1.02]
    best = None
    best_sum = -1.0

    for pattern in patterns:
        rows = len(pattern)
        max_cols = max(pattern)
        for s in s_vals:
            # compute nominal vertical step for hex pattern then apply small scales
            v0 = (np.sqrt(3)/2.0) * s
            for y_scale in y_scales:
                v = v0 * y_scale
                # Vertical centering
                total_h = (rows - 1) * v
                y0 = 0.5 - total_h / 2.0
                # try both offset parity choices (first row offset or not)
                for offset_start in (False, True):
                    centers = []
                    for r_idx, cnt in enumerate(pattern):
                        y = y0 + r_idx * v
                        row_width = (cnt - 1) * s if cnt > 1 else 0.0
                        x_start = 0.5 - row_width / 2.0
                        # determine offset for this row (hex-like)
                        offset = (r_idx % 2 == 1) if offset_start else (r_idx % 2 == 0)
                        if offset:
                            x_start += s/2.0
                        for j in range(cnt):
                            centers.append([x_start + j * s, y])
                    centers = np.array(centers[:n], dtype=float)
                    # small deterministic shifts to help edge packing
                    for dx,dy in ((0,0),(+0.006,0),( -0.006,0),(0,+0.006),(0,-0.006 )):
                        cand = centers + np.array([dx,dy])
                        cand = np.clip(cand, 1e-8, 1-1e-8)
                        radii = compute_max_radii(cand)
                        ssum = float(np.sum(radii))
                        if ssum > best_sum:
                            best_sum = ssum
                            best = (cand, radii, ssum)

    # fallback: if nothing found (shouldn't), return a central cluster
    if best is None:
        centers = np.array([[0.5,0.5]]*n)
        radii = compute_max_radii(centers)
        return centers, radii, float(np.sum(radii))

    return best

def compute_max_radii(centers, tol=1e-9, max_iter=3000):
    """
    Gauss-Seidel fixed point: r_i <- min(border_i, min_j (d_ij - r_j))
    Then perform conservative growth cycles and proportional fixes to
    exploit slack without creating overlaps.
    """
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    # border limits
    border = np.minimum.reduce([centers[:,0], centers[:,1], 1-centers[:,0], 1-centers[:,1]])
    border = np.maximum(border, 0.0)

    # pairwise distances
    dif = centers[:,None,:] - centers[None,:,:]
    d = np.sqrt(np.sum(dif*dif, axis=2))
    np.fill_diagonal(d, np.inf)

    # initialize radii conservatively at border distance
    r = border.copy()

    # initial shrink pass to remove overlaps (proportional scaling)
    for _ in range(1200):
        moved = False
        # wall constraints
        over = r - border
        if np.any(over > tol):
            r = np.minimum(r, border)
            moved = True
        # pairwise resolution
        for i in range(n):
            ri = r[i]
            for j in range(i+1, n):
                dij = d[i,j]
                if not np.isfinite(dij) or dij <= tol:
                    continue
                rj = r[j]
                if ri + rj > dij + tol:
                    # scale both proportionally (if both > 0)
                    denom = ri + rj
                    if denom <= 0:
                        # if both zero skip
                        continue
                    scale = dij / denom
                    if scale < 1.0 - 1e-12:
                        r[i] = ri * scale
                        r[j] = rj * scale
                        ri = r[i]
                        moved = True
        if not moved:
            break

    # Gauss-Seidel fixed-point to converge to maximal feasible radii
    for it in range(max_iter):
        prev = r.copy()
        for i in range(n):
            limits = d[i,:] - r
            limits[i] = np.inf
            # min over neighbor limits and border
            val = min(border[i], np.min(limits))
            r[i] = max(val, 0.0)
        if np.max(np.abs(r - prev)) < tol:
            break

    # Conservative growth cycles: increase a fraction of slack, then re-fix
    for cycle in range(8):
        # slack by walls and neighbors
        slack_wall = border - r
        slack_pair = np.min(d - (r[None,:] + r[:,None]), axis=1)
        slack_pair = np.maximum(slack_pair, 0.0)
        allowed_inc = np.minimum(slack_wall, slack_pair)
        allowed_inc = np.maximum(allowed_inc, 0.0)
        delta = 0.25 * allowed_inc
        if np.all(delta <= tol):
            break
        r += delta
        # quick proportional fix to restore feasibility
        for _ in range(300):
            moved = False
            # pairwise check
            for i in range(n):
                for j in range(i+1, n):
                    dij = d[i,j]
                    if not np.isfinite(dij):
                        continue
                    if r[i] + r[j] > dij + 1e-12:
                        scale = dij / (r[i] + r[j])
                        r[i] *= scale
                        r[j] *= scale
                        moved = True
            # wall enforcement
            over = r - border
            if np.any(over > 1e-12):
                r = np.minimum(r, border)
                moved = True
            if not moved:
                break

    # final safety clamp and non-negativity
    r = np.maximum(np.minimum(r, border), 0.0)
    return r

# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.6, edgecolor="k", linewidth=0.6)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualize:
    # visualize(centers, radii)