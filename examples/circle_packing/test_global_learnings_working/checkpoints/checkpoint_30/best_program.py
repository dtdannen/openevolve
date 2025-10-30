# EVOLVE-BLOCK-START
"""Constructor for 26-circle packing: layered hex-like layout + Gauss-Seidel radius maximizer.
Presents a deterministic explicit placement of centers (no global optimizer), but scans a
small parameter grid (margin, vertical scale, stagger) to mitigate edge effects and picks
the best total radius produced by a Gauss-Seidel-style radii maximizer.
"""
import numpy as np
from math import sqrt

def construct_packing():
    n = 26
    cols_per_row = [5, 6, 5, 5, 5]  # 26
    rows = len(cols_per_row)
    sqrt3 = sqrt(3.0)

    def make_centers(margin, vscale, stag):
        # available horizontal length between margins
        W = 1.0 - 2.0 * margin
        if W <= 0: 
            return None
        # pick a reference horizontal step using the widest row (to get near-hex)
        maxcnt = max(cols_per_row)
        if maxcnt > 1:
            step = W / (maxcnt - 1)
        else:
            step = W
        h = step * sqrt3 / 2.0 * vscale
        total_h = h * (rows - 1)
        y0 = 0.5 - total_h / 2.0
        centers = []
        for r, cnt in enumerate(cols_per_row):
            # center the row horizontally
            row_width = (cnt - 1) * step if cnt > 1 else 0.0
            x0 = 0.5 - row_width / 2.0
            xs = x0 + np.arange(cnt) * step
            # apply stagger on odd rows
            if r % 2 == 1:
                xs = xs + stag * step
            # clip into [margin,1-margin]
            xs = np.clip(xs, margin, 1.0 - margin)
            y = y0 + r * h
            for x in xs:
                centers.append((x, y))
        C = np.array(centers, dtype=float)
        if C.shape[0] != n:
            return None
        # reject if rows exceed vertical bounds
        if C[:,1].min() < -1e-12 or C[:,1].max() > 1.0 + 1e-12:
            return None
        return C

    def maximize_radii(C, tol=1e-10, max_iter=2000):
        nC = C.shape[0]
        border = np.minimum.reduce([C[:,0], C[:,1], 1.0 - C[:,0], 1.0 - C[:,1]]).astype(float)
        # pairwise distances
        diff = C[:, None, :] - C[None, :, :]
        d = np.sqrt((diff**2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        r = border.copy()  # init
        for it in range(max_iter):
            prev = r.copy()
            # Gauss-Seidel style: tighten each radius to its feasible upper bound
            for i in range(nC):
                ub = border[i]
                # d_ij - r_j for all j
                vals = d[i] - r
                vals[i] = np.inf
                ub = min(ub, float(np.min(vals)))
                r[i] = max(0.0, ub)
            if np.max(np.abs(r - prev)) < tol:
                break
        # final safety: correct tiny pairwise overlaps by proportional scaling per violating pair
        for i in range(nC):
            for j in range(i+1, nC):
                dij = d[i,j]
                if not np.isfinite(dij) or dij <= 1e-12:
                    continue
                if r[i] + r[j] > dij:
                    s = dij / (r[i] + r[j])
                    r[i] *= s
                    r[j] *= s
        # clamp to border again
        r = np.minimum(r, border)
        r = np.maximum(r, 0.0)
        return r

    best_sum = -1.0
    best_C = None
    best_r = None

    # parameter scan ranges (kept small and deterministic)
    for margin in np.linspace(0.02, 0.12, 19):
        for vscale in np.linspace(0.92, 1.08, 5):
            for stag in np.linspace(-0.25, 0.35, 7):
                C = make_centers(margin, vscale, stag)
                if C is None:
                    continue
                r = maximize_radii(C)
                s = float(r.sum())
                if s > best_sum:
                    best_sum = s
                    best_C, best_r = C, r

    # fallback if nothing found (shouldn't happen)
    if best_C is None:
        best_C = make_centers(0.06, 1.0, 0.0)
        best_r = maximize_radii(best_C)

    return best_C, best_r, float(best_r.sum())

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
        circle = Circle(center, radius, alpha=0.6, edgecolor="k")
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualize:
    # visualize(centers, radii)