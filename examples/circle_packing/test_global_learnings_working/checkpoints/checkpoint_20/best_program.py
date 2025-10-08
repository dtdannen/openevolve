# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles (margin search + Gauss-Seidel radius maximizer)"""
import numpy as np

def construct_packing():
    """
    Build 26 circle centers in a staggered near-hex layout and maximize radii
    for fixed centers using a Gauss-Seidel style solver that enforces:
      r_i <= border_i,  r_i <= d_ij - r_j  for all j!=i
    We sample several margins to mitigate edge effects and pick the best sum.
    """
    n = 26
    cols_per_row = [5, 6, 5, 5, 5]  # sums to 26
    rows = len(cols_per_row)

    def make_centers(margin):
        available_w = 1.0 - 2.0 * margin
        if available_w <= 0:
            return None
        centers = []
        # compute horizontal steps per row, then choose vertical hex-like step
        steps = []
        for cnt in cols_per_row:
            if cnt > 1:
                steps.append(available_w / (cnt - 1))
        if len(steps) == 0:
            avg_step = 0.5 * available_w
        else:
            avg_step = float(np.mean(steps))
        h = avg_step * np.sqrt(3.0) / 2.0 if avg_step > 0 else 0.0

        # vertical placement centered
        total_h = h * (rows - 1)
        y0 = 0.5 - total_h / 2.0
        for r, cnt in enumerate(cols_per_row):
            if cnt == 1:
                xs = np.array([0.5])
            else:
                xs = margin + np.linspace(0.0, available_w, cnt)
            if r % 2 == 1 and cnt > 1:
                step = xs[1] - xs[0]
                xs = xs + 0.5 * step
            xs = np.clip(xs, margin, 1.0 - margin)
            y = y0 + r * h
            for x in xs:
                centers.append((x, y))
        C = np.array(centers)
        # if any center lies outside [0,1], reject this margin
        if (C.min() < 0.0) or (C.max() > 1.0):
            return None
        return C

    def maximize_radii(C, tol=1e-9, max_iter=500):
        n = C.shape[0]
        # border limits
        border = np.minimum.reduce([C[:,0], C[:,1], 1.0 - C[:,0], 1.0 - C[:,1]]).astype(float)
        # pairwise distances
        diff = C[:, None, :] - C[None, :, :]
        dists = np.sqrt((diff**2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)

        # initial radii: border-limited
        r = border.copy()
        for it in range(max_iter):
            prev = r.copy()
            # Gauss-Seidel pass: set each r_i to its tightest feasible upper bound
            for i in range(n):
                ub = border[i]
                # compute d_ij - r_j for all j != i
                vals = dists[i] - r
                vals[i] = np.inf
                ub = min(ub, float(np.min(vals)))
                r[i] = max(0.0, ub)
            if np.max(np.abs(r - prev)) < tol:
                break

        # final safety: fix any slight overlaps by proportional scaling per violating pair
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i,j]
                if d <= 1e-12:
                    r[i] = r[j] = 0.0
                elif r[i] + r[j] > d:
                    s = d / (r[i] + r[j])
                    r[i] *= s
                    r[j] *= s
        return r

    best_sum = -1.0
    best_C = None
    best_r = None
    # sample margins (denser near small margins)
    for margin in np.linspace(0.02, 0.12, 21):
        C = make_centers(margin)
        if C is None or C.shape[0] != n:
            continue
        r = maximize_radii(C)
        s = float(r.sum())
        if s > best_sum:
            best_sum = s
            best_C = C
            best_r = r

    # fallback: if none found (shouldn't happen), use a conservative center layout
    if best_C is None:
        margin = 0.06
        best_C = make_centers(margin)
        best_r = maximize_radii(best_C)

    return best_C, best_r, float(np.sum(best_r))
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