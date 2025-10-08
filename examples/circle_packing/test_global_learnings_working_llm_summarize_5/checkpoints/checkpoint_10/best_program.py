# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using a truncated hexagonal lattice.
Places a dense cluster of 26 centers (closest to square center) from a hex grid
with spacing chosen to produce compact packing, then assigns each radius as the
minimum of distance to the unit-square border and half the nearest-neighbor distance.
This explicit constructor maximizes radii for fixed centers (no iterative search).
"""
import numpy as np

def construct_packing():
    n = 26

    # Hex lattice spacing chosen to give a dense cluster whose per-circle radius ~ spacing/2 ~ 0.1
    s = 0.20  # horizontal spacing
    v = s * np.sqrt(3) / 2.0  # vertical spacing for hex packing

    pts = []
    y = 0.0
    row = 0
    # generate hex lattice points inside [0,1] x [0,1]
    while y <= 1.0 + 1e-12:
        x_offset = (s / 2.0) if (row % 2 == 1) else 0.0
        x = x_offset
        while x <= 1.0 + 1e-12:
            # clamp numerical border overshoot
            if -1e-12 <= x <= 1 + 1e-12 and -1e-12 <= y <= 1 + 1e-12:
                pts.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
            x += s
        row += 1
        y += v

    pts = np.array(pts)
    if pts.shape[0] < n:
        # fallback: regular grid if hex grid too sparse
        m = int(np.ceil(np.sqrt(n)))
        xs = np.linspace(0.0, 1.0, m)
        ys = np.linspace(0.0, 1.0, m)
        grid = np.array([[x, y] for y in ys for x in xs])
        pts = grid

    # choose the n points closest to the center (densest patch)
    center = np.array([0.5, 0.5])
    dists = np.sum((pts - center) ** 2, axis=1)
    idx = np.argsort(dists)[:n]
    centers = pts[idx]

    # Slight jitter removal: re-center cluster by shifting average to exact center
    avg = centers.mean(axis=0)
    shift = center - avg
    centers = centers + shift
    centers = np.clip(centers, 0.0, 1.0)  # ensure inside unit square

    radii = compute_max_radii(centers)

    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii

def compute_max_radii(centers):
    """Assign radius to each center = min(dist to border, 0.5 * nearest neighbor distance)."""
    n = centers.shape[0]
    radii = np.zeros(n)
    # distance to borders
    border_dists = np.minimum.reduce([centers[:,0], centers[:,1], 1.0 - centers[:,0], 1.0 - centers[:,1]])
    # pairwise distances
    dmat = np.sqrt(np.maximum(0.0, ((centers[:,np.newaxis,:] - centers[np.newaxis,:,:])**2).sum(axis=2)))
    # replace diagonal with +inf so min excludes self
    np.fill_diagonal(dmat, np.inf)
    nearest = np.min(dmat, axis=1)
    # radius is min(border_dist, half nearest neighbor distance)
    radii = np.minimum(border_dists, 0.5 * nearest)
    # ensure non-negative
    radii = np.maximum(0.0, radii)
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

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualize:
    # visualize(centers, radii)