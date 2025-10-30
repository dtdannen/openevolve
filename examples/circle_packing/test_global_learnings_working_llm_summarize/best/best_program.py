# EVOLVE-BLOCK-START
"""Explicit constructor for 26 circles in the unit square.

We place 26 centers in a hex-like layered layout (rows [3,5,5,5,5,3]) but
then compute radii by maximizing the sum subject to non-overlap and border
constraints using a simple iterative repair (greedy) that enforces
r_i + r_j <= dist(i,j) and r_i <= border_dist_i. This lets edge circles
grow larger than the conservative equal-radius design and typically raises
the total sum of radii.
"""
import numpy as np


def construct_packing():
    rows = [3, 5, 5, 5, 5, 3]  # sums to 26
    n = sum(rows)
    assert n == 26

    sqrt3 = np.sqrt(3.0)
    R = len(rows)
    max_in_row = max(rows)

    # A conservative design spacing to place centers (hex-like)
    r_horiz = 1.0 / (2.0 * max_in_row)
    r_vert = 1.0 / (((R - 1) * sqrt3) + 2.0)
    r_design = min(r_horiz, r_vert)

    centers = []
    for k, count in enumerate(rows):
        # alternate offset for approximate hex packing
        offset = (k % 2) * r_design
        total_width = 2.0 * r_design * (count - 1) if count > 1 else 0.0
        # center the row horizontally, then apply small parity offset
        x_start = 0.5 - 0.5 * total_width + offset
        # clamp start so first center is at least r_design from left
        x_start = max(x_start, r_design)
        xs = [x_start + j * 2.0 * r_design for j in range(count)]
        y = r_design + k * (sqrt3 * r_design)
        for x in xs:
            centers.append([x, y])

    centers = np.array(centers, dtype=float)
    assert centers.shape == (n, 2)

    # pairwise distances
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)

    # distance to border: a circle cannot have radius greater than its
    # minimum distance to the four sides
    border_dists = np.minimum.reduce([centers[:, 0], centers[:, 1], 1.0 - centers[:, 0], 1.0 - centers[:, 1]])

    # Start with radii at border distances (this lets edge circles be large).
    radii = border_dists.copy()

    # Iteratively enforce pairwise non-overlap: r_i + r_j <= d_ij
    # Greedy repair: when a pair violates, reduce the larger radius by exactly
    # the overlap amount. Repeat until all constraints satisfied (or max iters).
    tol = 1e-12
    max_iters = 20000
    for it in range(max_iters):
        max_change = 0.0
        # Randomize order slightly to avoid cycling deterministically
        idxs = list(range(n))
        # go through pairs; deterministic ordering is fine as n small
        for i in range(n):
            ri = radii[i]
            # vectorized check against all j>i
            sums = ri + radii[i + 1 :]
            dij = dists[i, i + 1 :]
            # find violations
            viol = np.where(sums > dij + tol)[0]
            if viol.size == 0:
                continue
            for vv in viol:
                j = i + 1 + int(vv)
                overlap = (radii[i] + radii[j]) - dists[i, j]
                if overlap <= tol:
                    continue
                # reduce the larger of the two by the overlap (minimally correct)
                if radii[i] >= radii[j]:
                    new_ri = max(0.0, radii[i] - overlap)
                    max_change = max(max_change, abs(new_ri - radii[i]))
                    radii[i] = new_ri
                else:
                    new_rj = max(0.0, radii[j] - overlap)
                    max_change = max(max_change, abs(new_rj - radii[j]))
                    radii[j] = new_rj
        # enforce border caps (in case pairwise reductions allowed increasing elsewhere)
        new_radii = np.minimum(radii, border_dists)
        max_change = max(max_change, np.max(np.abs(new_radii - radii)))
        radii = new_radii
        if max_change < 1e-10:
            break
    else:
        # final pass: project any remaining violations (rare)
        for i in range(n):
            for j in range(i + 1, n):
                if radii[i] + radii[j] > dists[i, j] + 1e-12:
                    excess = radii[i] + radii[j] - dists[i, j]
                    # split excess reduction proportional to current radii (avoid zero divide)
                    s = radii[i] + radii[j]
                    if s <= 0:
                        radii[i] = radii[j] = 0.0
                    else:
                        radii[i] = max(0.0, radii[i] - excess * (radii[i] / s))
                        radii[j] = max(0.0, radii[j] - excess * (radii[j] / s))
        radii = np.minimum(radii, border_dists)

    # safety floor
    radii = np.maximum(radii, 0.0)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii


# EVOLVE-BLOCK-END


def run_packing():
    """Run the circle packing constructor for n=26"""
    return construct_packing()


def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    for i, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(c, r, alpha=0.6, edgecolor="k"))
        ax.text(c[0], c[1], str(i), ha="center", va="center", fontsize=8)
    plt.title(f"n={len(centers)}, sum radii={sum(radii):.6f}")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # visualize(centers, radii)