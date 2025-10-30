# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles (pattern search over explicit layouts)"""
import numpy as np

def construct_packing():
    """
    Build several deterministic hex-like layouts (different row patterns),
    scan a small range of horizontal spacings s and pick the best resulting
    packing (max sum of radii). This remains a constructive method: we
    place centers by formula and only perform a deterministic radius projection.
    """
    n = 26
    # Several plausible hex-like row patterns summing to 26
    patterns = [
        [5, 6, 5, 5, 5],
        [6, 5, 5, 5, 5],
        [6, 5, 6, 5, 4],
        [5, 6, 6, 5, 4],
    ]

    best = None
    best_sum = -1.0

    # modest grid of spacings to try (center-to-center horizontally)
    s_vals = np.linspace(0.13, 0.20, 15)

    for rows in patterns:
        R = len(rows)
        maxk = max(rows)
        for s in s_vals:
            h = np.sqrt(3) / 2 * s
            # center vertically the block in [0,1]
            total_h = (R - 1) * h
            y0 = 0.5 - total_h / 2.0
            centers = []
            for r_idx, k in enumerate(rows):
                y = y0 + r_idx * h
                row_width = (k - 1) * s
                x_start = 0.5 - row_width / 2.0
                # stagger alternate rows for hex adjacency
                if r_idx % 2 == 1:
                    x_start += s / 2.0
                for j in range(k):
                    centers.append([x_start + j * s, y])
            centers = np.array(centers[:n], dtype=float)
            # Slightly clip to avoid exact-border degeneracies
            centers = np.clip(centers, 1e-6, 1 - 1e-6)

            radii = compute_max_radii(centers)
            ssum = float(np.sum(radii))
            if ssum > best_sum:
                best_sum = ssum
                best = (centers, radii, ssum)

    # Fallback: if nothing found (shouldn't happen), return a simple pattern
    if best is None:
        centers = np.array([[0.5,0.5]]*n)
        radii = compute_max_radii(centers)
        return centers, radii, float(np.sum(radii))

    return best

def compute_max_radii(centers, tol=1e-9):
    """
    Deterministic projection to feasible radii:
      - start with wall distances
      - repeatedly fix pairwise overlaps by proportional shrinking
      - apply a few conservative fractional growth passes to use slack
    """
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    # distance to wall
    wall = np.minimum.reduce([centers[:,0], centers[:,1], 1-centers[:,0], 1-centers[:,1]])
    # start with wall-limited radii
    r = wall.copy()

    # pairwise distances
    dif = centers[:,None,:] - centers[None,:,:]
    d = np.sqrt(np.sum(dif*dif, axis=2))
    np.fill_diagonal(d, np.inf)

    # Shrink overlaps: proportional scaling until no violations
    for _ in range(1500):
        moved = False
        # ensure wall constraints
        over = r - wall
        if np.any(over > tol):
            r = np.minimum(r, wall)
            moved = True
        # pairwise
        for i in range(n):
            ri = r[i]
            # vectorized check for j>i could be done, but keep simple deterministic loop
            for j in range(i+1, n):
                dij = d[i,j]
                if not np.isfinite(dij) or dij <= tol:
                    continue
                rj = r[j]
                if ri + rj > dij + tol:
                    # scale both proportionally
                    scale = dij / (ri + rj)
                    if scale < 1.0 - 1e-12:
                        r[i] = ri * scale
                        r[j] = rj * scale
                        ri = r[i]  # update local var
                        moved = True
        if not moved:
            break

    # Conservative growth cycles: small fraction of available slack
    for cycle in range(10):
        # available increase by walls
        inc_wall = wall - r
        # available by neighbors: for each i, min_j (d_ij - (r_i + r_j)) + r_i => max increase = min_j(d_ij - (r_i + r_j))
        avail = np.full(n, np.inf)
        # compute pairwise slack matrix
        slack = d - (r[None,:] + r[:,None])
        # for each i, the maximum increase without immediate overlap is min(slack[i,:])
        avail = np.minimum(avail, np.min(slack, axis=1))
        # overall allowed inc is min(inc_wall, avail)
        max_inc = np.minimum(inc_wall, avail)
        max_inc = np.maximum(max_inc, 0.0)
        # apply a cautious fraction
        delta = 0.3 * max_inc
        if np.all(delta <= tol):
            break
        r += delta
        # re-shrink a few iterations to remove any slight violations
        for _ in range(200):
            moved = False
            # enforce pairwise
            for i in range(n):
                for j in range(i+1, n):
                    dij = d[i,j]
                    if not np.isfinite(dij) or dij <= tol:
                        continue
                    if r[i] + r[j] > dij + tol:
                        scale = dij / (r[i] + r[j])
                        r[i] *= scale
                        r[j] *= scale
                        moved = True
            # wall enforcement
            over = r - wall
            if np.any(over > tol):
                r = np.minimum(r, wall)
                moved = True
            if not moved:
                break

    r = np.maximum(r, 0.0)
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
        circle = Circle(center, radius, alpha=0.6, ec="k")
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualize:
    # visualize(centers, radii)