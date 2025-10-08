# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles in a unit square.
Creates an explicit near-hexagonal layered layout (row counts [6,5,5,5,5])
and tunes the lattice spacing to maximize the sum of radii (simple grid search).
"""
import numpy as np


def construct_packing():
    n = 26
    # row counts that sum to 26: a near-hexagonal layering
    rows = [6, 5, 5, 5, 5]

    def build_centers(s):
        """Build centers for lattice spacing s (horizontal spacing)."""
        v = np.sqrt(3) / 2 * s  # vertical spacing for hex arrangement
        centers = []
        R = len(rows)
        total_v = (R - 1) * v
        y0 = (1.0 - total_v) / 2.0
        for r_idx, k in enumerate(rows):
            span = (k - 1) * s
            x0 = (1.0 - span) / 2.0
            # alternate shift for hex packing
            shift = (r_idx % 2) * (s / 2.0)
            y = y0 + r_idx * v
            for i in range(k):
                x = x0 + i * s + shift
                centers.append([x, y])
        return np.array(centers)

    def radii_for_centers(centers):
        """Compute non-overlapping radii: limited by walls and half nearest neighbor."""
        m = centers.shape[0]
        # distance to walls
        wall = np.minimum.reduce([centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]])
        # pairwise distances
        radii = wall.copy()
        for i in range(m):
            # compute distances to others
            dists = np.sqrt(np.sum((centers - centers[i]) ** 2, axis=1))
            dists[i] = np.inf
            min_dist = np.min(dists)
            radii[i] = min(radii[i], 0.5 * min_dist)
        return radii

    # Search for the best horizontal spacing s that maximizes sum of radii
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # sensible bounds: too small -> tiny circles; too large -> centers outside square
    # pick grid of spacings
    for s in np.linspace(0.06, 0.24, 200):
        centers = build_centers(s)
        # reject if any center lies outside [0,1] (numerical safety)
        if np.any(centers < 0 - 1e-8) or np.any(centers > 1 + 1e-8):
            continue
        radii = radii_for_centers(centers)
        # if any radius non-positive, skip
        if np.any(radii <= 0):
            continue
        ssum = float(np.sum(radii))
        if ssum > best_sum:
            best_sum = ssum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Fallback: if search failed, place a simple layout
    if best_centers is None:
        best_centers = build_centers(0.12)
        best_radii = radii_for_centers(best_centers)
        best_sum = float(np.sum(best_radii))

    return best_centers, best_radii, best_sum


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