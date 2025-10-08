# EVOLVE-BLOCK-START
"""Improved constructor-based circle packing for n=26 circles.

This uses a hexagonal-like layered layout (rows with alternating offsets)
to pack 26 circles in a unit square with equal radii chosen to be as large
as possible given the row with the maximum count. This yields a much larger
sum of radii than the original simple concentric-ring layout.
"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Row layout chosen to sum to 26: 6 + 5 + 6 + 5 + 4 = 26
    rows = [6, 5, 6, 5, 4]
    n = sum(rows)
    assert n == 26

    # Spacing factors for hexagonal-style packing
    sqrt3 = np.sqrt(3.0)
    max_in_row = max(rows)

    # Horizontal constraint: 2 * r * max_in_row <= 1  -> r <= 1/(2*max_in_row)
    r_horiz = 1.0 / (2.0 * max_in_row)

    # Vertical constraint for rows: last_center_y <= 1 - r
    # if y_k = r + k * (sqrt3 * r) for k=0..(R-1), require r + (R-1)*sqrt3*r <= 1 - r
    # -> r * ((R-1)*sqrt3 + 2) <= 1
    R = len(rows)
    r_vert = 1.0 / (((R - 1) * sqrt3) + 2.0)

    # Choose the limiting radius (horizontal typically limits for our layout)
    r = min(r_horiz, r_vert)

    centers = []
    # Build rows from bottom (k=0) to top (k=R-1)
    for k, count in enumerate(rows):
        # Alternate offset every other row (hex packing)
        offset = (k % 2) == 1
        if not offset:
            x_start = r  # centers: r, r+2r, r+4r, ...
        else:
            x_start = 2.0 * r  # shifted by r: 2r, 4r, ...
        # horizontal spacing 2r
        xs = [x_start + j * 2.0 * r for j in range(count)]
        y = r + k * (sqrt3 * r)
        for x in xs:
            centers.append([x, y])

    centers = np.array(centers, dtype=float)
    # Safety: ensure count and containment
    assert centers.shape == (n, 2)

    # Use equal radii (designed to be non-overlapping inside the unit square)
    radii = np.full(n, r, dtype=float)

    # Validate and (if numeric tiny issues) correct radii so no overlaps and inside bounds
    # Compute minimal distance to borders and to neighbors, then clamp to those maxima.
    # This is a conservative correction; for a correctly computed r it leaves values unchanged.
    # Distance to border
    border_dists = np.minimum.reduce([centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]])
    max_by_border = border_dists.copy()

    # Minimal half pairwise distances
    pairwise = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    # set diagonal to large so min ignores zero
    np.fill_diagonal(pairwise, np.inf)
    min_half_dist = 0.5 * pairwise.min(axis=1)

    # Final allowed radius per circle
    allowed = np.minimum(max_by_border, min_half_dist)
    # Numerical tolerance: do not enlarge beyond our design r, only shrink if necessary
    radii = np.minimum(radii, allowed)
    # If any radius became extremely small (shouldn't happen), fall back to allowed values
    radii = np.maximum(radii, 1e-12)

    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


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

    plt.title(f"Hex-style Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualize:
    # visualize(centers, radii)