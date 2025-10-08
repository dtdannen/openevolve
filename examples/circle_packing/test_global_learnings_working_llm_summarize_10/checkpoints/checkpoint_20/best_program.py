# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles (deterministic hex cluster + staged refinement)"""
import numpy as np
from itertools import product


def construct_packing():
    """
    Build a deterministic layout of 26 circle centers based on a compact
    hex-like cluster (rows 5+5+6+5+5). Use a coarse uniform scale search
    to find a good global packing, then refine with small anisotropic
    tweaks and per-row horizontal shifts to break perfect symmetry.
    Returns centers (26,2), radii (26,), sum_radii
    """
    n = 26
    # Compact hex-like row counts summing to 26
    rows = [5, 5, 6, 5, 5]

    # Base integer layout coordinates (unit horizontal spacing)
    base = []
    vstep = np.sqrt(3) / 2.0  # hex vertical step for unit horiz spacing
    for r_idx, cnt in enumerate(rows):
        xs = np.arange(cnt) - (cnt - 1) / 2.0
        x_offset = 0.5 * (r_idx % 2)
        for xi in xs:
            base.append([xi + x_offset, r_idx * vstep])
    base = np.array(base)  # (26,2)

    # bounding box of base pattern
    minb = base.min(axis=0)
    maxb = base.max(axis=0)
    width = maxb[0] - minb[0]
    height = maxb[1] - minb[1]

    # small margin to avoid numerical border-touch
    margin = 1e-4
    # maximum uniform scale to fit inside [margin,1-margin]
    Smax = min((1 - 2 * margin) / width, (1 - 2 * margin) / height)

    def compute_radii(centers):
        # distance to border
        d_border = np.minimum.reduce([centers[:, 0], centers[:, 1], 1 - centers[:, 0], 1 - centers[:, 1]])
        # pairwise distances
        C = centers
        dif = C[:, None, :] - C[None, :, :]
        D = np.sqrt(np.maximum(0.0, (dif ** 2).sum(axis=2)))
        np.fill_diagonal(D, np.inf)
        nearest = D.min(axis=1)
        radii = np.minimum(d_border, 0.5 * nearest)
        radii = np.maximum(radii, 0.0)
        return radii

    best_sum = -1.0
    best_centers = None
    best_radii = None
    best_s = None

    # Coarse uniform scale sweep
    for alpha in np.linspace(0.40, 1.00, 61):
        s = alpha * Smax
        scaled = (base - minb) * s
        # center cluster inside unit square
        scaled_min = scaled.min(axis=0)
        scaled_max = scaled.max(axis=0)
        translate = 0.5 - 0.5 * (scaled_min + scaled_max)
        centers = scaled + translate
        centers = np.clip(centers, margin, 1 - margin)
        radii = compute_radii(centers)
        ssum = radii.sum()
        if ssum > best_sum:
            best_sum = ssum
            best_centers = centers.copy()
            best_radii = radii.copy()
            best_s = s

    # Refinement: small anisotropic scaling around best_s and per-row horizontal shifts
    # Per-row shifts (in units of current spacing s) to break symmetry; try -0.03,0,0.03 fractions
    shift_fracs = [-0.03, 0.0, 0.03]
    tx_vals = [0.996, 1.0, 1.004]
    ty_vals = [0.996, 1.0, 1.004]

    # Precompute base x positions per row for convenience
    row_indices = []
    idx = 0
    for cnt in rows:
        row_indices.append(list(range(idx, idx + cnt)))
        idx += cnt
    # horizontal spacing in base units is 1.0, so shift in absolute units = frac * best_s
    for tx, ty in product(tx_vals, ty_vals):
        s_x = best_s * tx
        s_y = best_s * ty * (np.sqrt(3) / 2.0) / (np.sqrt(3) / 2.0)  # keep vertical ratio consistent
        # build scaled base with these anisotropic scales
        anis_base = base.copy()
        # apply anisotropic scale: scale x by (s_x), y by (s_y / vstep) so that unit vertical spacing becomes s_y
        anis_base[:, 0] = (anis_base[:, 0] - minb[0]) * (s_x)
        anis_base[:, 1] = (anis_base[:, 1] - minb[1]) * (s_y / vstep)
        # try all combinations of small per-row horizontal shifts (3^5 = 243)
        for shifts in product(shift_fracs, repeat=len(rows)):
            centers = anis_base.copy()
            # apply per-row x shift to each member in row (shift is fraction of best_s)
            for r, shift_frac in enumerate(shifts):
                inds = row_indices[r]
                centers[inds, 0] += shift_frac * best_s
            # center into unit square
            cmin = centers.min(axis=0)
            cmax = centers.max(axis=0)
            translate = 0.5 - 0.5 * (cmin + cmax)
            centers = centers + translate
            centers = np.clip(centers, margin, 1 - margin)
            radii = compute_radii(centers)
            ssum = radii.sum()
            if ssum > best_sum:
                best_sum = ssum
                best_centers = centers.copy()
                best_radii = radii.copy()

    # As a final small local tweak, try translating the whole pattern to snug against edges slightly
    # (keeps deterministic)
    for dx, dy in [(0.0, 0.0), (-1e-3, 0.0), (1e-3, 0.0), (0.0, -1e-3), (0.0, 1e-3)]:
        centers = np.clip(best_centers + np.array([dx, dy]), margin, 1 - margin)
        radii = compute_radii(centers)
        ssum = radii.sum()
        if ssum > best_sum:
            best_sum = ssum
            best_centers = centers.copy()
            best_radii = radii.copy()

    return np.array(best_centers), np.array(best_radii), float(best_radii.sum())


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
        circle = Circle(center, radius, alpha=0.6, edgecolor="k", linewidth=0.6)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # Uncomment to visualize:
    # visualize(centers, radii)