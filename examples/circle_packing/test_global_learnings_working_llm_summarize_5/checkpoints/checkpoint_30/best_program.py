# EVOLVE-BLOCK-START
"""Deterministic constructor for packing n=26 circles in the unit square.

Strategy:
- Generate truncated hexagonal lattices for a small set of horizontal spacings s.
- For each spacing, take the 26 lattice points closest to the square center to form a dense patch.
- For the chosen patch compute radii by r_i = min(dist_to_border, 0.5*nearest_neighbor_dist).
- Pick the spacing that yields the largest sum of radii and return those centers/radii.

This explicit constructor is deterministic and aims to reproduce the compact,
near-hexagonal cluster that yields high total radius sum for n=26.
"""
import numpy as np

def construct_packing():
    n = 26
    center = np.array([0.5, 0.5])
    # candidate horizontal spacings to try (covers a range around good hex spacings)
    spacings = np.linspace(0.15, 0.24, 10)  # tuned range; deterministic

    best_sum = -1.0
    best_centers = None
    best_radii = None

    sqrt3 = np.sqrt(3.0)
    for s in spacings:
        v = s * sqrt3 / 2.0  # vertical spacing for hex lattice
        pts = []
        # Generate enough rows/columns so that we have many lattice points around the unit square
        # Row index range chosen to cover slightly beyond [0,1]
        # Start y slightly negative to capture staggered rows
        y = -v
        row = 0
        while y <= 1.0 + v:
            x_offset = (s / 2.0) if (row % 2 == 1) else 0.0
            x = -s + x_offset
            while x <= 1.0 + s:
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    pts.append((x, y))
                x += s
            row += 1
            y += v

        pts = np.array(pts)
        if pts.shape[0] < n:
            # fallback regular grid if too few points (shouldn't happen for chosen spacings)
            m = int(np.ceil(np.sqrt(n)))
            xs = np.linspace(0.0, 1.0, m)
            ys = np.linspace(0.0, 1.0, m)
            pts = np.array([[x, y] for y in ys for x in xs])

        # choose the n points whose squared distance to center is smallest
        d2 = np.sum((pts - center) ** 2, axis=1)
        idx = np.argsort(d2)[:n]
        centers = pts[idx].copy()

        # Shift cluster so its centroid matches the square center (helps symmetry)
        avg = centers.mean(axis=0)
        shift = center - avg
        centers += shift
        # Clip to remain inside the unit square (numerical safety)
        centers = np.clip(centers, 0.0, 1.0)

        radii = compute_radii_centers(centers)
        ssum = float(np.sum(radii))
        if ssum > best_sum:
            best_sum = ssum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # final safety: ensure shapes and sum returned
    if best_centers is None:
        # degrade gracefully: use a simple symmetric hex-like pattern
        best_centers = make_fallback_centers(n)
        best_radii = compute_radii_centers(best_centers)
        best_sum = float(np.sum(best_radii))

    return best_centers, best_radii, float(best_sum)

def compute_radii_centers(centers):
    """Compute radii as min(distance to border, 0.5*nearest_neighbor_distance)."""
    n = centers.shape[0]
    xs = centers[:, 0]; ys = centers[:, 1]
    border_dists = np.minimum.reduce([xs, ys, 1.0 - xs, 1.0 - ys])

    # pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dmat = np.sqrt(np.maximum(0.0, (diff ** 2).sum(axis=2)))
    # exclude self by setting diagonal to +inf
    np.fill_diagonal(dmat, np.inf)
    nearest = np.min(dmat, axis=1)

    radii = np.minimum(border_dists, 0.5 * nearest)
    radii = np.maximum(0.0, radii)
    return radii

def make_fallback_centers(n):
    """Fallback symmetric hex-like layout summing to n (used only in degenerate case)."""
    # pattern 4,6,6,6,4 = 26
    pattern = [4,6,6,6,4]
    max_k = max(pattern)
    a_x = 1.0 / (max_k + 1.0)
    a_y = a_x * np.sqrt(3.0) / 2.0
    rows = len(pattern)
    centers = []
    y0 = 0.5 - (rows - 1) / 2.0 * a_y
    for r, k in enumerate(pattern):
        y = y0 + r * a_y
        x_start = 0.5 - (k - 1) / 2.0 * a_x
        for c in range(k):
            x = x_start + c * a_x
            centers.append([x, y])
    centers = np.array(centers[:n])
    centers = np.clip(centers, 1e-8, 1 - 1e-8)
    return centers

# EVOLVE-BLOCK-END

# Fixed external interface
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

# Optional simple visualizer (not used in automated eval)
def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    for c, r in zip(centers, radii):
        ax.add_patch(Circle(c, r, alpha=0.6, edgecolor='k', linewidth=0.5))
    plt.show()

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(centers, radii)