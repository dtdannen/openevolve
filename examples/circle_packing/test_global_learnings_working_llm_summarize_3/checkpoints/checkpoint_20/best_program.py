# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using a hexagonal lattice constructor."""
import numpy as np


def construct_packing():
    """
    Construct a deterministic arrangement of 26 circle centers in [0,1]^2
    using a hexagonal (offset-grid) lattice. We choose the largest lattice
    spacing s that still produces at least 26 candidate centers, then
    keep the 26 centers closest to the square center to mitigate edge effects.
    Radii are set to the maximal non-overlapping values given these centers:
        r_i = min(dist_to_border(i), 0.5 * min_j dist(center_i, center_j))
    This yields a valid non-overlapping packing and tends to produce dense,
    hexagonal-like local arrangements that maximize the sum of radii.
    Returns:
        (centers (26,2), radii (26,), sum_of_radii)
    """
    n_target = 26

    def make_hex_centers(s):
        # margin so circles can have radius up to s/2 without crossing border
        margin = s / 2.0
        pts = []
        y = margin
        row = 0
        y_step = s * np.sqrt(3) / 2.0
        while y <= 1.0 - margin + 1e-12:
            x_offset = (row % 2) * (s / 2.0)
            x = margin + x_offset
            while x <= 1.0 - margin + 1e-12:
                pts.append((x, y))
                x += s
            row += 1
            y += y_step
        return np.array(pts)

    # Binary search for the largest s that still yields at least n_target centers.
    low, high = 0.01, 0.6  # high > possible spacing
    best_s = low
    for _ in range(40):
        mid = 0.5 * (low + high)
        pts = make_hex_centers(mid)
        if pts.shape[0] >= n_target:
            best_s = mid
            low = mid  # we can try larger spacing
        else:
            high = mid
    centers_all = make_hex_centers(best_s)

    # If we have more than needed, keep the n_target centers closest to square center
    if centers_all.shape[0] > n_target:
        center_of_square = np.array([0.5, 0.5])
        dists = np.linalg.norm(centers_all - center_of_square, axis=1)
        idx = np.argsort(dists)[:n_target]
        centers = centers_all[idx]
    else:
        centers = centers_all.copy()

    # As a small enhancement, ensure we always return exactly n_target centers
    # (should be guaranteed by construction/trimming above)
    if centers.shape[0] < n_target:
        # fallback: fill with evenly spaced points along a small spiral near center
        extra = n_target - centers.shape[0]
        spiral = []
        for k in range(extra):
            t = 0.05 * (k + 1)
            angle = 2 * np.pi * (k + 1) / extra
            spiral.append([0.5 + t * np.cos(angle), 0.5 + t * np.sin(angle)])
        centers = np.vstack([centers, np.array(spiral)])

    # Sort centers for determinism (by x then y)
    centers = centers[np.lexsort((centers[:, 1], centers[:, 0]))]

    # Compute maximal valid radii:
    radii = compute_max_radii(centers)

    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    For a fixed set of centers, compute a safe maximal radius assignment
    that guarantees no overlaps and containment in the unit square.
    We use the conservative but tight rule:
        r_i = min(dist_to_border(i), 0.5 * min_j dist(i,j))
    which makes touching allowed but not overlapping.
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    # distance to borders
    dist_border = np.minimum.reduce([centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]])
    # pairwise distances (compute efficiently)
    diff = centers[:, None, :] - centers[None, :, :]  # n x n x 2
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # set diagonal to a large number to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists, axis=1)
    # radii limited by half the nearest neighbor distance and by borders
    radii = np.minimum(dist_border, 0.5 * min_pair)
    # Ensure non-negative and tiny numerical floor
    radii = np.maximum(radii, 0.0)
    return radii


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
        circle = Circle(center, radius, alpha=0.6, edgecolor='k', linewidth=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Hex-lattice Constructor Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualize:
    # visualize(centers, radii)