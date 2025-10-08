# EVOLVE-BLOCK-START
"""Improved deterministic constructor for packing n=26 circles in the unit square.

Design:
- Use a compact hex-like layered layout tuned for 26 circles: rows [6,5,6,5,4].
- Compute maximal feasible radii for fixed centers via a stable, vectorized
  iterative relaxation that (a) resolves overlaps by proportional shrinking
  using the most restrictive pairwise scale per circle, (b) enforces wall limits,
  and (c) attempts controlled growth toward available slack to increase the sum
  of radii.
Returns centers (26,2), radii (26,), sum_radii.
"""
import numpy as np

def construct_packing():
    n = 26
    # compact hex-like rows summing to 26
    rows = [6, 5, 6, 5, 4]
    R = len(rows)
    max_k = max(rows)

    # choose horizontal spacing so widest row nearly fills the unit width
    margin = 1e-6
    if max_k > 1:
        s = (1.0 - 2 * margin) / (max_k - 1)
    else:
        s = 0.5
    h = s * np.sqrt(3.0) / 2.0

    # if vertical span would exceed unit square, scale down spacing
    span_v = (R - 1) * h
    if span_v > 1.0 - 2 * margin:
        h = (1.0 - 2 * margin) / (R - 1)
        s = h * 2.0 / np.sqrt(3.0)

    # build centered rows deterministically
    y0 = 0.5 - ((R - 1) * h) / 2.0
    centers = []
    for i, k in enumerate(rows):
        y = y0 + i * h
        left = 0.5 - ((k - 1) * s) / 2.0
        xs = left + np.arange(k) * s
        for x in xs:
            centers.append([float(x), float(y)])
    centers = np.array(centers[:n], dtype=float)

    radii = compute_max_radii(centers)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii

def compute_max_radii(centers, max_iters=800, tol=1e-10):
    n = centers.shape[0]
    # initial radii limited by walls
    border = np.minimum.reduce([centers[:,0], centers[:,1],
                                1.0 - centers[:,0], 1.0 - centers[:,1]])
    radii = border.copy()

    if n <= 1:
        return radii

    # precompute pairwise distances
    diffs = centers[:, None, :] - centers[None, :, :]
    D = np.hypot(diffs[...,0], diffs[...,1])
    np.fill_diagonal(D, np.inf)

    for it in range(max_iters):
        old = radii.copy()

        # resolve overlaps in a vectorized way:
        ri = radii[:, None]
        rj = radii[None, :]
        sumr = ri + rj
        viol = sumr > D
        if np.any(viol):
            # per-pair scale factors (<=1). For non-violations factor=1
            factors = np.ones_like(sumr)
            factors[viol] = D[viol] / sumr[viol]
            # for each circle take the most restrictive (minimum) scale across its pairs
            min_scale = factors.min(axis=1)
            # avoid exact zeros
            min_scale = np.maximum(min_scale, 1e-12)
            radii *= min_scale

        # enforce walls
        radii = np.minimum(radii, border)

        # attempt controlled growth toward available slack:
        # allowed_i = min(border_i, min_j (D_ij - r_j))
        slack = D - radii[None, :]  # allowed center-to-center remaining for each pair
        allowed = np.minimum(border, slack.min(axis=1))
        allowed = np.maximum(allowed, 0.0)
        grow_mask = allowed > radii + 1e-14
        radii[grow_mask] = (radii[grow_mask] + allowed[grow_mask]) / 2.0

        if np.max(np.abs(radii - old)) < tol:
            break

    radii = np.maximum(radii, 0.0)
    return radii

# EVOLVE-BLOCK-END

def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

# Optional visualization helper (not used in automated evaluation)
def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    for c, r in zip(centers, radii):
        ax.add_patch(Circle(c, r, alpha=0.6, edgecolor='k'))
    plt.show()

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")