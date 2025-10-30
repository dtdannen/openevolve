# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles (deterministic multi-candidate constructor)"""
import numpy as np

def construct_packing():
    n = 26
    center = np.array([0.5, 0.5])

    # candidate generators: hex lattice (vary spacing, rotation, compression)
    s_vals = np.linspace(0.105, 0.235, 18)
    thetas = [0.0, np.pi/18, np.pi/12]  # small rotations
    alphas = [0.80, 0.88, 0.94, 1.0]    # compress toward center to fight edge effects

    best_sum = -1.0
    best_centers = None
    best_radii = None

    # prepare integer grid
    M = 9
    ii = np.arange(-M, M + 1)
    jj = np.arange(-M, M + 1)
    I, J = np.meshgrid(ii, jj)
    IJ = np.column_stack((I.ravel(), J.ravel()))

    for s in s_vals:
        # hex basis before rotation
        v1 = np.array([s, 0.0])
        v2 = np.array([s * 0.5, s * (np.sqrt(3) / 2.0)])
        base_pts = IJ.dot(np.vstack((v1, v2)).T)  # large grid
        for theta in thetas:
            c = np.cos(theta); si = np.sin(theta)
            R = np.array([[c, -si], [si, c]])
            pts_rot = base_pts.dot(R.T) + center  # rotate and center
            # keep only points roughly inside square (with small margin)
            mask = np.all((pts_rot >= -1e-9) & (pts_rot <= 1 + 1e-9), axis=1)
            valid = pts_rot[mask]
            if valid.shape[0] < n:
                continue
            # sort by distance to center and take a local cloud
            d2 = np.sum((valid - center) ** 2, axis=1)
            order = np.argsort(d2)
            cloud = valid[order[: max(n + 12, 2 * n)]]

            for alpha in alphas:
                # compress slightly toward center
                offsets = cloud - center
                comp = center + offsets * alpha
                comp = np.clip(comp, 1e-9, 1 - 1e-9)
                # pick closest n to center
                d2c = np.sum((comp - center) ** 2, axis=1)
                pick = np.argsort(d2c)[:n]
                centers = comp[pick]
                radii = compute_max_radii_relax(centers, max_iter=1200, tol=1e-8)
                ssum = float(np.sum(radii))
                if ssum > best_sum:
                    best_sum = ssum
                    best_centers = centers.copy()
                    best_radii = radii.copy()

    # also try a few structured row patterns (hand-tuned)
    row_patterns = [
        [5,5,4,4,4,4],  # compact
        [6,5,5,5,5],    # wider middle
        [5,6,5,5,5],    # alternative
    ]
    for rows in row_patterns:
        max_cols = max(rows)
        # try few spacings
        for s in np.linspace(0.12, 0.20, 9):
            sin60 = np.sqrt(3)/2
            dy = s * sin60
            # compute vertical offset to center
            num_rows = len(rows)
            height = (num_rows - 1) * dy
            y0 = 0.5 - height / 2.0
            centers_list = []
            for ri, k in enumerate(rows):
                offset = (s/2.0) if (ri % 2 == 1) else 0.0
                row_width = (k - 1) * s
                x0 = 0.5 - row_width / 2.0 + offset
                y = y0 + ri * dy
                for j in range(k):
                    x = x0 + j * s
                    centers_list.append([x, y])
            centers_arr = np.clip(np.array(centers_list), 1e-9, 1-1e-9)
            if centers_arr.shape[0] != n:
                continue
            radii = compute_max_radii_relax(centers_arr, max_iter=1200, tol=1e-8)
            ssum = float(np.sum(radii))
            if ssum > best_sum:
                best_sum = ssum
                best_centers = centers_arr.copy()
                best_radii = radii.copy()

    # final fallback: a symmetric concentric pattern
    if best_centers is None:
        best_centers = np.zeros((n,2))
        best_centers[0] = center
        for i in range(1,9):
            ang = 2*np.pi*(i-1)/8
            best_centers[i] = center + 0.18*np.array([np.cos(ang), np.sin(ang)])
        for i in range(16):
            ang = 2*np.pi*i/16
            best_centers[i+9] = center + 0.34*np.array([np.cos(ang), np.sin(ang)])
        best_centers = np.clip(best_centers, 1e-9, 1-1e-9)
        best_radii = compute_max_radii_relax(best_centers)

    sum_radii = float(np.sum(best_radii))
    return best_centers, best_radii, sum_radii

def compute_max_radii_relax(centers, max_iter=800, tol=1e-7):
    """
    Gauss-Seidel projection: for fixed centers compute maximal feasible radii.
    r_i := min(dist_to_wall_i, min_j (dist_ij - r_j))
    """
    n = centers.shape[0]
    border = np.minimum.reduce([centers[:,0], centers[:,1], 1-centers[:,0], 1-centers[:,1]])
    r = border.copy()
    # precompute distances
    diff = centers[:,None,:] - centers[None,:,:]
    D = np.sqrt(np.maximum(np.sum(diff*diff,axis=2), 1e-12))
    for it in range(max_iter):
        maxc = 0.0
        for i in range(n):
            # compute candidate bounds dist_ij - r_j
            cand = D[i] - r
            cand[i] = border[i]  # ignore self
            # only non-negative allowed
            allowed = np.minimum(border[i], np.min(np.maximum(cand, 0.0)))
            change = abs(r[i] - allowed)
            if change > 0:
                r[i] = allowed
                if change > maxc: maxc = change
        if maxc < tol:
            break
    return np.maximum(0.0, r)

# EVOLVE-BLOCK-END

# fixed runner
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    for c,r in zip(centers, radii):
        ax.add_patch(Circle(c, r, alpha=0.5, edgecolor='k'))
    plt.show()

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s:.6f}")
    # visualize(centers, radii)