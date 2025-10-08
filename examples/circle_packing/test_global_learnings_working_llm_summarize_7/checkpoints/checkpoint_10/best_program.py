# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles (explicit hex-like constructor)"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii using a
    hexagonal-like lattice with row counts tuned to 26.
    Returns:
        Tuple of (centers, radii, sum_of_radii)
    """
    n = 26
    # Hex-like design parameters: choose spacing s as large as possible for 6 columns
    s = 1.0 / 6.0  # horizontal spacing: max for 6 columns
    h = s * np.sqrt(3) / 2.0  # vertical spacing for hex lattice

    # Row structure summing to 26
    rows = [6, 5, 6, 5, 4]  # total 26
    R = len(rows)

    # vertical placement centered in [0,1]
    y0 = 0.5 - ( (R - 1) * h ) / 2.0

    centers = []
    for i, k in enumerate(rows):
        y = y0 + i * h
        # center each row horizontally
        left = 0.5 - ((k - 1) * s) / 2.0
        xs = left + np.arange(k) * s
        for x in xs:
            centers.append([x, y])

    centers = np.array(centers[:n])  # ensure length 26

    # Compute conservative maximal radii:
    # For each circle, radius <= distance to border, and <= 0.5 * min_pairwise_distance
    radii = np.empty(n)
    for i in range(n):
        x, y = centers[i]
        border_dist = min(x, y, 1 - x, 1 - y)
        # compute min distance to any other center
        if n == 1:
            min_pair = 2.0  # irrelevant
        else:
            dists = np.sqrt(((centers - centers[i]) ** 2).sum(axis=1))
            dists[i] = np.inf
            min_pair = dists.min()
        # take half the nearest neighbor distance (guarantees non-overlap) and border limit
        radii[i] = min(border_dist, 0.5 * min_pair)

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