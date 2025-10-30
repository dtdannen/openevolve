# EVOLVE-BLOCK-START
"""Deterministic explicit constructor for packing n=26 circles in the unit square.

This version improves on prior attempts by:
- Trying a compact family of asymmetric hex-like row patterns that sum to 26.
- Scanning a fine grid of horizontal spacings and slight vertical scalings.
- Trying both stagger parities and small global shifts to exploit edge slack.
- Computing maximal feasible radii for fixed centers using a robust procedure:
    * initialize at wall-distances
    * proportional shrink passes to remove overlaps
    * Gauss-Seidel fixed-point to converge to a feasible maximal set
    * cautious growth cycles with proportional repairs to capture slack
All steps are deterministic and constructor-only (no random search).
"""
import numpy as np

def construct_packing():
    n = 26
    # Diverse plausible row patterns summing to 26 (break symmetry to help edges)
    patterns = [
        [5, 6, 5, 5, 5],
        [6, 5, 5, 5, 5],
        [6, 5, 6, 5, 4],
        [5, 6, 6, 5, 4],
        [4, 6, 6, 5, 5],
    ]

    # search parameters tuned for good coverage but modest eval time
    s_vals = np.linspace(0.125, 0.195, 30)   # horizontal center-to-center spacing
    y_scales = [0.94, 0.98, 1.00, 1.02]      # slight vertical compression/expansion
    shifts = [(0.0, 0.0), (0.006, 0.0), (-0.006, 0.0), (0.0, 0.006), (0.0, -0.006)]

    best_sum = -1.0
    best_centers = None
    best_radii = None

    for pattern in patterns:
        rows = len(pattern)
        max_cols = max(pattern)
        for s in s_vals:
            # ideal hex vertical step for spacing s
            v0 = (np.sqrt(3.0) / 2.0) * s
            for y_scale in y_scales:
                v = v0 * y_scale
                # compute total vertical span and center block vertically
                total_h = (rows - 1) * v if rows > 1 else 0.0
                if total_h > 1.0 + 1e-12:
                    continue
                y0 = 0.5 - total_h / 2.0
                for offset_start in (False, True):
                    centers = []
                    for r_idx, cnt in enumerate(pattern):
                        y = y0 + r_idx * v
                        row_span = (cnt - 1) * s if cnt > 1 else 0.0
                        x_start = 0.5 - row_span / 2.0
                        offset = (r_idx % 2 == 1) if offset_start else (r_idx % 2 == 0)
                        if offset:
                            x_start += s / 2.0
                        # ensure row stays inside [0,1] by tiny shift if necessary
                        last_x = x_start + (cnt - 1) * s
                        if x_start < 0.0:
                            x_start = 0.0
                        if last_x > 1.0:
                            x_start -= (last_x - 1.0)
                        for j in range(cnt):
                            centers.append([x_start + j * s, y])
                    # trim/pad to exactly n deterministically
                    centers = np.array(centers, dtype=float)
                    if centers.shape[0] < n:
                        # pad with small positions along bottom-left corner
                        needed = n - centers.shape[0]
                        extra_xs = np.linspace(0.02, 0.18, needed)
                        extras = np.vstack([extra_xs, np.full(needed, 0.02)]).T
                        centers = np.vstack((centers, extras))
                    elif centers.shape[0] > n:
                        centers = centers[:n]

                    # try small global shifts to exploit slack near walls
                    for dx, dy in shifts:
                        cand = centers + np.array([dx, dy])
                        cand = np.clip(cand, 1e-9, 1.0 - 1e-9)
                        radii = compute_max_radii(cand)
                        ssum = float(np.sum(radii))
                        if ssum > best_sum + 1e-12:
                            best_sum = ssum
                            best_centers = cand.copy()
                            best_radii = radii.copy()

    # fallback single layout (should not be needed)
    if best_centers is None:
        rows = [5, 6, 5, 5, 5]
        s = 0.16
        v = (np.sqrt(3.0) / 2.0) * s
        y0 = 0.5 - (len(rows) - 1) * v / 2.0
        centers = []
        for r_idx, cnt in enumerate(rows):
            y = y0 + r_idx * v
            row_span = (cnt - 1) * s if cnt > 1 else 0.0
            x_start = 0.5 - row_span / 2.0
            if r_idx % 2 == 1:
                x_start += s / 2.0
            for j in range(cnt):
                centers.append([x_start + j * s, y])
        centers = np.clip(np.array(centers[:n], dtype=float), 1e-9, 1.0 - 1e-9)
        best_centers = centers
        best_radii = compute_max_radii(centers)
        best_sum = float(np.sum(best_radii))

    return best_centers, best_radii, float(best_sum)


def compute_max_radii(centers, tol=1e-10, max_iter=3000):
    """
    Robust deterministic procedure to compute near-maximal radii for fixed centers:
      - initialize at border distances
      - proportional shrink passes to remove overlaps
      - Gauss-Seidel fixed-point to converge
      - cautious growth cycles with proportional repairs to capture slack
    """
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # distances to walls
    border = np.minimum.reduce([centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]])
    border = np.maximum(border, 0.0)

    # pairwise distances
    dif = centers[:, None, :] - centers[None, :, :]
    d = np.sqrt(np.sum(dif * dif, axis=2))
    np.fill_diagonal(d, np.inf)

    # initialize radii at border distances (feasible)
    r = border.copy()

    # Initial proportional shrink to remove any overlaps caused by numerical issues
    for _ in range(1000):
        moved = False
        # enforce walls
        over = r - border
        if np.any(over > 0):
            r = np.minimum(r, border)
            moved = True
        # pairwise proportional fixes
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                dij = d[i, j]
                if not np.isfinite(dij) or dij <= 0.0:
                    continue
                rj = r[j]
                if ri + rj > dij + 1e-12:
                    total = ri + rj
                    if total <= 0.0:
                        r[i] = r[j] = 0.0
                        moved = True
                        continue
                    scale = dij / total
                    # apply scale only if it reduces
                    if scale < 1.0 - 1e-14:
                        r[i] = ri * scale
                        r[j] = rj * scale
                        ri = r[i]
                        moved = True
        if not moved:
            break

    # Gauss-Seidel fixed-point to approach maximal feasible radii
    for it in range(max_iter):
        prev = r.copy()
        for i in range(n):
            limits = d[i, :] - r
            limits[i] = np.inf
            val = np.min(limits)
            # enforce wall limit
            if val > border[i]:
                val = border[i]
            if val < 0.0:
                val = 0.0
            r[i] = val
        if np.max(np.abs(r - prev)) < tol:
            break

    # Conservative growth cycles: use small fraction of slack, then repair
    for cycle in range(10):
        # slack to walls
        slack_wall = border - r
        # slack to neighbors: for each i, min_j (d_ij - (r_i + r_j))
        slack_pair = np.min(d - (r[None, :] + r[:, None]), axis=1)
        slack_pair = np.maximum(slack_pair, 0.0)
        allowed_inc = np.minimum(slack_wall, slack_pair)
        allowed_inc = np.maximum(allowed_inc, 0.0)
        delta = 0.22 * allowed_inc  # cautious fraction
        if np.all(delta <= 1e-12):
            break
        r += delta

        # proportional repair loops to restore feasibility
        for _ in range(300):
            moved = False
            # pairwise check
            for i in range(n):
                for j in range(i + 1, n):
                    dij = d[i, j]
                    if not np.isfinite(dij):
                        continue
                    if r[i] + r[j] > dij + 1e-12:
                        total = r[i] + r[j]
                        if total <= 0.0:
                            r[i] = r[j] = 0.0
                            moved = True
                            continue
                        scale = dij / total
                        r[i] *= scale
                        r[j] *= scale
                        moved = True
            # walls
            over = r - border
            if np.any(over > 1e-12):
                r = np.minimum(r, border)
                moved = True
            if not moved:
                break

    # final safety clamps
    r = np.maximum(np.minimum(r, border), 0.0)
    return r

# EVOLVE-BLOCK-END


# fixed runner API
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    for c, rr in zip(centers, radii):
        ax.add_patch(Circle(c, rr, alpha=0.6, edgecolor="k"))
    plt.title(f"n={len(centers)}, sum radii={sum(radii):.6f}")
    plt.show()


if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # To visualize, uncomment:
    # visualize(centers, radii)