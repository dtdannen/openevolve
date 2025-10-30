# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles using a compact hex-like layout.

This implementation:
- Builds several candidate hexagonal/row layouts for 26 circles (rows 3..7).
- For each candidate it computes a conservative maximal uniform radius that fits
  rows horizontally and vertically inside the unit square.
- Chooses the candidate with largest uniform radius (hence largest initial sum).
- Converts that layout to explicit centers and then runs a robust iterative
  relaxation to compute the maximal feasible radii for those fixed centers
  (solving r_i + r_j <= d_ij and r_i <= border_i by fixed-point iteration).
This yields a much higher sum of radii than the original naive rings.
"""
import numpy as np

def construct_packing():
    n = 26

    # Helper: compute conservative vertical radius limit for given number of rows
    sqrt3 = np.sqrt(3.0)
    def vertical_limit(rows):
        if rows == 1:
            return 0.5
        # top and bottom margins r each: r + (rows-1)*sqrt3*r <= 1 - r
        # => r * ( (rows-1)*sqrt3 + 2) <= 1
        return 1.0 / (((rows - 1) * sqrt3) + 2.0)

    # Helper: conservative horizontal limit for a row with m columns.
    # If the row is offset (shifted by r horizontally), we use a slightly
    # tighter bound 1/(2m+1), otherwise 1/(2m). These are conservative but safe.
    def horizontal_limit(m, offset):
        if offset:
            return 1.0 / (2 * m + 1.0)
        else:
            return 1.0 / (2 * m)

    best_score = -1.0
    best_layout = None  # (rows, counts_list, offset_start, r_candidate)

    # Try different numbers of rows and simple distributions of 26 items:
    for rows in range(3, 8):  # 3..7 rows (reasonable for 26)
        base = n // rows
        rem = n % rows
        # Two simple ways to distribute remainder: prefix or suffix
        for prefix_heavy in (True, False):
            if prefix_heavy:
                counts = [base + 1 if i < rem else base for i in range(rows)]
            else:
                counts = [base if i < (rows - rem) else base + 1 for i in range(rows)]
            # Try both choices of whether the first row is offset or not
            for offset_start in (False, True):
                # compute conservative r limit
                r_v = vertical_limit(rows)
                r_hs = []
                for i, m in enumerate(counts):
                    offset = (i % 2 == 1) if offset_start else (i % 2 == 0)
                    r_hs.append(horizontal_limit(m, offset))
                r_candidate = min(r_v, min(r_hs))
                # Score is sum of radii equal to n * r_candidate (uniform)
                score = n * r_candidate
                if score > best_score:
                    best_score = score
                    best_layout = (rows, counts, offset_start, r_candidate)

    # Build centers from chosen layout
    rows, counts, offset_start, r = best_layout
    centers = []
    vstep = np.sqrt(3.0) * r
    total_height = 0.0 if rows == 1 else (rows - 1) * vstep
    y0 = 0.5 - total_height / 2.0
    for i in range(rows):
        m = counts[i]
        offset = (i % 2 == 1) if offset_start else (i % 2 == 0)
        row_span = 0.0 if m == 1 else (m - 1) * 2.0 * r
        # centered start
        start_x = 0.5 - row_span / 2.0
        if offset:
            # shift by +r for offset rows (hex packing)
            start_x += r
        # ensure numeric safety: clamp start_x so centers lie within [r, 1-r]
        start_x = max(start_x, r)
        # but also ensure last center doesn't exceed 1-r; if it does, nudge left
        last_x = start_x + (m - 1) * 2.0 * r
        if last_x > 1.0 - r:
            start_x -= (last_x - (1.0 - r))

        y = y0 + i * vstep
        for k in range(m):
            x = start_x + k * 2.0 * r
            centers.append([x, y])

    centers = np.array(centers, dtype=float)
    # Defensive: if count mismatch (shouldn't), trim or pad with small-offset positions
    if centers.shape[0] > n:
        centers = centers[:n]
    elif centers.shape[0] < n:
        # add extra small circles near corners if needed
        needed = n - centers.shape[0]
        extras = np.array([[0.02 + 0.96 * (i / max(1, needed - 1)), 0.02] for i in range(needed)])
        centers = np.vstack((centers, extras))

    # Compute maximal feasible radii for these fixed centers via iterative relaxation
    radii = compute_max_radii(centers)

    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


def compute_max_radii(centers, max_iter=2000, tol=1e-8):
    """
    Compute maximal radii satisfying:
        r_i <= border_i
        r_i + r_j <= dist(i,j)  for all i != j

    We perform a Gauss-Seidel style fixed-point iteration:
        r_i <- min( border_i, min_j (dist_ij - r_j) )
    which is monotone and converges to the maximal feasible solution.
    """
    n = centers.shape[0]
    # border distances
    b = np.minimum.reduce([centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]])
    b = np.maximum(b, 0.0)

    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    # avoid self-dist zeros causing issues; self distance won't be used except as inf
    np.fill_diagonal(d, np.inf)

    # initialize radii at border distances (feasible relative to walls)
    rads = b.copy()

    # Iterate: r_i = min(b_i, min_j (d_ij - r_j))
    for it in range(max_iter):
        prev = rads.copy()
        # Gauss-Seidel update: update in-place using newest values immediately
        for i in range(n):
            limits = d[i, :] - rads  # d_ij - r_j (using latest r_j)
            # We must ignore self (set to +inf)
            limits[i] = np.inf
            r_new = min(b[i], np.min(limits))
            # ensure non-negative
            if r_new < 0.0:
                r_new = 0.0
            rads[i] = r_new
        delta = np.max(np.abs(rads - prev))
        if delta < tol:
            break

    # Final small clean: ensure pairwise constraints numerically satisfied
    # If any tiny violations exist, reduce involved radii proportionally.
    for i in range(n):
        for j in range(i + 1, n):
            dij = d[i, j]
            if rads[i] + rads[j] > dij + 1e-12:
                # reduce proportionally
                scale = dij / (rads[i] + rads[j])
                rads[i] *= scale
                rads[j] *= scale

    # clamp by borders again (safety)
    rads = np.minimum(rads, b)
    rads = np.maximum(rads, 0.0)
    return rads
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