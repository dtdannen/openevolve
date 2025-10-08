# EVOLVE-BLOCK-START
"""Improved constructor-based circle packing for n=26 using a maximal hex-grid constructor
with small alignment search, then per-center maximal-radius propagation."""
import numpy as np

def construct_packing():
    """
    Build a near-optimal deterministic arrangement of 26 circles in the unit square.
    Strategy:
     - Try several hex-grid alignments (horizontal offset 0 or r, small vertical shifts)
     - For each alignment, binary-search the largest equal radius r that yields >=26
       grid centers inside [r,1-r].
     - Select the best alignment (largest r), take the first 26 grid centers,
       then run a constraint-propagation routine to maximize individual radii
       (respecting border and pairwise non-overlap). This allows radii to vary
       and increases the total sum beyond 26*r.
    Returns (centers, radii, sum_of_radii).
    """
    n_needed = 26
    sqrt3 = np.sqrt(3.0)

    def gen_grid(r, horiz_offset, vert_shift):
        pts = []
        y = r + vert_shift
        row = 0
        while y <= 1.0 - r + 1e-12:
            offset = (row % 2) * r + horiz_offset
            # place x from r+offset to 1-r
            x = r + offset
            while x <= 1.0 - r + 1e-12:
                if r - 1e-12 <= x <= 1.0 - r + 1e-12:
                    pts.append((x, y))
                x += 2.0 * r
            y += sqrt3 * r
            row += 1
        return np.array(pts)

    def max_r_for_alignment(horiz_offset, vert_frac_steps=3):
        # binary search for largest r such that count >= n_needed for some vert_shift
        lo, hi = 1e-8, 0.5
        best_r = 0.0
        best_pts = None
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            found = False
            for frac in np.linspace(0.0, 0.5, vert_frac_steps):
                vert_shift = frac * sqrt3 * mid
                pts = gen_grid(mid, horiz_offset, vert_shift)
                if pts.shape[0] >= n_needed:
                    found = True
                    break
            if found:
                best_r = mid
                lo = mid
            else:
                hi = mid
        # Reconstruct best point set with best_r and choose lexicographically first n_needed
        if best_r > 0:
            # pick best vertical shift that gives most symmetric placement (centered)
            best_pts = None
            for frac in np.linspace(0.0, 0.5, vert_frac_steps):
                vert_shift = frac * sqrt3 * best_r
                pts = gen_grid(best_r, horiz_offset, vert_shift)
                if pts.shape[0] >= n_needed:
                    # sort by y descending (center-first) then x ascending for stability
                    pts_sorted = pts[np.lexsort((pts[:, 0], -pts[:, 1]))]
                    if best_pts is None or pts_sorted.shape[0] > best_pts.shape[0]:
                        best_pts = pts_sorted
            if best_pts is None:
                best_pts = gen_grid(best_r, horiz_offset, 0.0)
        return best_r, best_pts

    # Try two horizontal offsets (0 or r) to allow denser packings
    best_global = {"r": 0.0, "pts": None}
    for horiz_choice in (0.0, 1.0):  # 1.0 stands for 'r' later scaled inside function
        # horiz_offset is either 0.0 or r -> we encode r by passing 0.0 or 'r' marker
        # To keep gen_grid signature simple, convert marker to actual offset inside max_r_for_alignment
        if horiz_choice == 0.0:
            horiz_offset = 0.0
        else:
            # use a lambda-like trick: pass offset as 'r' marker, handle by passing mid in gen call
            # We'll handle by trying horiz_offset = mid inside max_r_for_alignment by calling twice:
            horiz_offset = None  # indicator
        # Custom version to handle horiz_offset == None (meaning offset=r)
        def max_r_custom(h_offset_flag):
            lo, hi = 1e-8, 0.5
            best_r = 0.0
            best_pts = None
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                found = False
                for frac in np.linspace(0.0, 0.5, 4):
                    vert_shift = frac * sqrt3 * mid
                    ho = (mid if h_offset_flag is None else h_offset_flag)
                    pts = gen_grid(mid, ho, vert_shift)
                    if pts.shape[0] >= n_needed:
                        found = True
                        break
                if found:
                    best_r = mid
                    lo = mid
                else:
                    hi = mid
            if best_r > 0:
                for frac in np.linspace(0.0, 0.5, 6):
                    vert_shift = frac * sqrt3 * best_r
                    ho = (best_r if h_offset_flag is None else h_offset_flag)
                    pts = gen_grid(best_r, ho, vert_shift)
                    if pts.shape[0] >= n_needed:
                        pts_sorted = pts[np.lexsort((pts[:, 0], -pts[:, 1]))]
                        if best_pts is None or pts_sorted.shape[0] > best_pts.shape[0]:
                            best_pts = pts_sorted
                if best_pts is None:
                    best_pts = gen_grid(best_r, (best_r if h_offset_flag is None else h_offset_flag), 0.0)
            return best_r, best_pts

        r_val, pts = max_r_custom(horiz_offset)
        if r_val > best_global["r"] and pts is not None:
            best_global["r"] = r_val
            best_global["pts"] = pts.copy()

    if best_global["pts"] is None:
        # Fallback to previous deterministic row layout (guarantees feasibility)
        rows = [6,5,6,5,4]
        sqrt3 = np.sqrt(3.0)
        max_k = max(rows)
        r_h = 1.0 / (2.0 * max_k)
        r_v = 1.0 / (2.0 + (len(rows) - 1) * sqrt3)
        base_r = min(r_h, r_v)
        vert_span = 2.0 * base_r + (len(rows) - 1) * base_r * sqrt3
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
        centers = centers[:n_needed]
    else:
        # pick the first n_needed sorted points
        pts = best_global["pts"]
        centers = pts[:n_needed].copy()
        # ensure all centers strictly within [0,1]
        centers = np.clip(centers, 1e-12, 1.0 - 1e-12)

    # Now compute maximal radii for these fixed centers
    radii = compute_max_radii(centers)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii

def compute_max_radii(centers, tol=1e-12, max_iters=2000):
    """
    Constraint-propagation to maximize sum of radii given fixed centers.
    Start from border-limited radii and greedily resolve pairwise overlaps by
    reducing the larger radius (this tends to minimize decrease in total sum).
    """
    n = centers.shape[0]
    # start with boundary-limited radii
    radii = np.minimum.reduce((
        centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 0],
        1.0 - centers[:, 1]
    )).astype(float)

    # Precompute pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    # Repeatedly enforce r_i + r_j <= d_ij by reducing the larger radius when needed.
    for it in range(max_iters):
        max_change = 0.0
        for i in range(n):
            ri = radii[i]
            # vectorized check against j>i
            # but keep simple loops for determinism
            for j in range(i + 1, n):
                d = dists[i, j]
                if d <= 0.0:
                    continue
                rj = radii[j]
                if ri + rj > d:
                    # reduce the larger radius only
                    if ri > rj:
                        new_ri = max(0.0, d - rj)
                        change = ri - new_ri
                        if change > 0:
                            radii[i] = new_ri
                            ri = new_ri
                            max_change = max(max_change, change)
                    else:
                        new_rj = max(0.0, d - ri)
                        change = rj - new_rj
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