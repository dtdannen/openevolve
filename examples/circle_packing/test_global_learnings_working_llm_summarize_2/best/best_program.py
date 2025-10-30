# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles (improved)"""
import numpy as np


def construct_packing():
    """
    Construct an explicit arrangement of 26 circle centers in the unit square
    using a hexagonal-like layered layout (rows: 5,6,5,6,4 = 26). Then compute
    maximal feasible radii by a Gauss-Seidel style relaxation so circles do not
    overlap and stay inside the unit square.

    Returns:
        (centers, radii, sum_of_radii)
    """
    n = 26
    # Define hex-like layout: rows counts sum to 26
    rows = [5, 6, 5, 6, 4]  # total 26
    R = len(rows)

    # Choose horizontal spacing s (depends on largest row length) and margin
    margin = 0.03
    max_row = max(rows)
    # ensure (max_row-1)*s + 2*margin <= 1  => s <= (1-2*margin)/(max_row-1)
    s = min(0.188, (1 - 2 * margin) / (max_row - 1))
    v = s * np.sqrt(3) / 2.0  # vertical step for hex packing

    # Compute vertical span and center it
    total_height = (R - 1) * v
    y0 = margin + (1 - 2 * margin - total_height) / 2.0

    centers = []
    for r, count in enumerate(rows):
        # center x of row so the row is centered in the square
        row_width = (count - 1) * s
        x0 = margin + (1 - 2 * margin - row_width) / 2.0
        y = y0 + r * v
        # For hex staggering: shift every other row by s/2
        shift = (s / 2.0) if (r % 2 == 1) else 0.0
        for k in range(count):
            x = x0 + shift + k * s
            centers.append([x, y])

    centers = np.array(centers[:n])
    # Clip to ensure strictly inside [margin,1-margin]
    centers = np.clip(centers, margin, 1 - margin)

    radii = compute_max_radii(centers)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def compute_max_radii(centers, max_iter=1000, tol=1e-8):
    """
    Compute maximal feasible radii for given centers using iterative relaxation:
    For each circle i, radius[i] = min(distance_to_border[i],
      min_j max(0, dist(i,j) - radius[j]))
    Iterate (Gauss-Seidel) until convergence.
    """
    n = centers.shape[0]
    # initial radii: distance to border
    border_dists = np.minimum.reduce([centers[:, 0], centers[:, 1],
                                      1 - centers[:, 0], 1 - centers[:, 1]])
    radii = border_dists.copy()

    # Precompute pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2)) + np.eye(n)  # add diag to avoid zero
    # iterative relaxation
    for it in range(max_iter):
        max_change = 0.0
        for i in range(n):
            # limit from neighbors: for each j != i, radius_i <= dist_ij - radius_j
            # compute minimal allowed radius from neighbors
            # ignore self by using dists[i,j] with j!=i
            allowed = border_dists[i]
            # vectorized compute candidate = dist_ij - radii[j]
            cand = dists[i, :] - radii
            # exclude self
            cand[i] = allowed
            # ensure nonnegative
            cand = np.maximum(cand, 0.0)
            allowed = min(allowed, np.min(cand[np.arange(n) != i]))
            # update
            newr = allowed
            max_change = max(max_change, abs(newr - radii[i]))
            radii[i] = newr
        if max_change < tol:
            break

    # final safety clamp
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