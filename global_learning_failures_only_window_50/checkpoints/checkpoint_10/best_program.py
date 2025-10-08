# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles (hex-like rows)"""
import numpy as np

def construct_packing():
    """
    Construct a deterministic arrangement of 26 circles in the unit square
    using a hexagonal-row layout: rows = [6,5,6,5,4]. Then compute maximal
    radii by enforcing boundary and pairwise non-overlap constraints
    via iterative pairwise adjustments.
    Returns (centers, radii, sum_of_radii).
    """
    # Layout: five rows summing to 26 circles
    rows = [6, 5, 6, 5, 4]
    m = len(rows)
    max_k = max(rows)

    # geometric spacings for ideal hex packing with equal radii r
    sqrt3 = np.sqrt(3.0)
    # limits from horizontal and vertical spans:
    r_h = 1.0 / (2.0 * max_k)                        # 2r*k <= 1 -> r <= 1/(2k)
    r_v = 1.0 / (2.0 + (m - 1) * sqrt3)              # 2r + (m-1)*r*sqrt3 <= 1
    base_r = min(r_h, r_v)

    # compute vertical and horizontal offsets to center the pattern
    vert_span = 2.0 * base_r + (m - 1) * base_r * sqrt3
    top = (1.0 - vert_span) / 2.0 + base_r
    dy = base_r * sqrt3

    centers = []
    for row_idx, k in enumerate(rows):
        row_span = 2.0 * base_r * k
        left = (1.0 - row_span) / 2.0 + base_r
        xs = left + np.arange(k) * (2.0 * base_r)
        y = top + row_idx * dy
        for x in xs:
            centers.append((x, y))

    centers = np.array(centers)
    # ensure we have 26
    assert centers.shape[0] == 26

    radii = compute_max_radii(centers)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def compute_max_radii(centers, tol=1e-10, max_iters=2000):
    """
    Given fixed centers inside [0,1]^2, compute radii that maximize sum of
    radii subject to: 0 <= r_i <= dist_to_border_i and r_i + r_j <= dist(i,j).
    We use a simple iterative constraint-propagation method:
    - initialize r_i = dist_to_border_i
    - repeatedly scan pairs; if r_i + r_j > d_ij reduce the larger of (r_i,r_j)
      to d_ij - the_other (this keeps the pair sum at most d_ij while not
      shrinking both unnecessarily).
    This converges in practice for well-separated hex-like layouts.
    """
    n = centers.shape[0]
    # start with boundary-limited radii
    radii = np.minimum(centers[:, 0], centers[:, 1])
    radii = np.minimum(radii, 1.0 - centers[:, 0])
    radii = np.minimum(radii, 1.0 - centers[:, 1])

    # Precompute pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid self-dist zero issues
    np.fill_diagonal(dists, np.inf)

    for it in range(max_iters):
        max_change = 0.0
        # scan pairs (i<j)
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d <= 0.0:
                    continue
                ri, rj = radii[i], radii[j]
                if ri + rj > d:
                    # reduce the larger radius to satisfy ri + rj <= d
                    if ri > rj:
                        new_ri = max(0.0, d - rj)
                        change = abs(new_ri - ri)
                        if change > 0:
                            radii[i] = new_ri
                            max_change = max(max_change, change)
                    else:
                        new_rj = max(0.0, d - ri)
                        change = abs(new_rj - rj)
                        if change > 0:
                            radii[j] = new_rj
                            max_change = max(max_change, change)
        if max_change < tol:
            break

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