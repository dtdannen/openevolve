# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles (hexagonal-layer constructor)"""
import numpy as np

def construct_packing():
    """
    Construct 26 circle centers in a layered hexagonal pattern inside the unit square.
    Centers are chosen according to row lengths [4,5,6,6,5] (total 26).
    We search for an inter-center spacing s that fits the pattern in the square
    and then choose the s that maximizes the sum of radii computed as
    min(distance to border, half nearest-neighbor distance).
    Returns centers (26,2), radii (26,), sum_radii.
    """
    rows = [4, 5, 6, 6, 5]  # sums to 26
    R = len(rows)

    def make_centers(s):
        h = s * np.sqrt(3) / 2.0
        centers = []
        y0 = 0.5 - (R - 1) * h / 2.0
        for r, L in enumerate(rows):
            y = y0 + r * h
            # alternate row offsets for hex pattern
            offset = (s / 2.0) if (r % 2 == 1) else 0.0
            x0 = 0.5 - (L - 1) * s / 2.0 + offset
            xs = x0 + np.arange(L) * s
            for x in xs:
                centers.append([x, y])
        return np.array(centers)

    def fits_in_square(s):
        C = make_centers(s)
        return (C.min() >= 0.0) and (C.max() <= 1.0)

    # binary search maximum spacing s that keeps centers inside unit square
    lo, hi = 1e-6, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fits_in_square(mid):
            lo = mid
        else:
            hi = mid
    s_max = lo

    # sample s values near s_max to maximize sum of radii (border vs neighbors tradeoff)
    sample_lo = max(1e-6, 0.5 * s_max)
    sample_hi = s_max
    samples = np.linspace(sample_lo, sample_hi, 60)

    def radii_and_sum_for_s(s):
        C = make_centers(s)
        # distance to border
        dist_border = np.minimum.reduce([C[:,0], C[:,1], 1 - C[:,0], 1 - C[:,1]])
        # pairwise distances
        dif = C[:, None, :] - C[None, :, :]
        dists = np.sqrt((dif**2).sum(axis=2)) + np.eye(C.shape[0])  # add 1 on diag to ignore zeros
        nearest = dists.min(axis=1)
        # half nearest neighbor distance (safe but slightly conservative)
        rad_from_neighbors = 0.5 * nearest
        radii = np.minimum(dist_border, rad_from_neighbors)
        radii = np.maximum(radii, 0.0)
        return C, radii, radii.sum()

    best_C, best_r, best_sum = None, None, -1.0
    for s in samples:
        C, r, ssum = radii_and_sum_for_s(s)
        if ssum > best_sum:
            best_C, best_r, best_sum = C, r, ssum

    # As a final small improvement, allow increasing some radii greedily where border allows:
    # (this preserves non-overlap because we recompute per-pair half distances)
    C = best_C.copy()
    # recompute precise half-pairwise minimums (exclude self by inf on diagonal)
    dif = C[:, None, :] - C[None, :, :]
    dists = np.sqrt((dif**2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    half_nearest = 0.5 * dists.min(axis=1)
    dist_border = np.minimum.reduce([C[:,0], C[:,1], 1 - C[:,0], 1 - C[:,1]])
    radii = np.minimum(dist_border, half_nearest)
    radii = np.maximum(radii, 0.0)
    sum_radii = radii.sum()

    return C, radii, float(sum_radii)


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
        circle = Circle(center, radius, alpha=0.6, edgecolor='k')
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualize:
    # visualize(centers, radii)