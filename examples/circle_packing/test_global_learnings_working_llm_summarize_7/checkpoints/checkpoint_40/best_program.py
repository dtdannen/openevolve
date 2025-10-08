# EVOLVE-BLOCK-START
"""Deterministic constructor for n=26 circle packing with a small
center-adjustment pass to improve the hand-crafted layouts.

Approach:
 - Build 3 geometry-inspired candidate center sets (hex-rows, rings, staggered).
 - Use a robust deterministic radius relaxation to get non-overlapping radii.
 - Pick the best candidate by sum of radii, then run a short deterministic
   greedy center-adjustment pass (small discrete moves) to improve the sum.
No randomness; all choices are deterministic.
"""
import numpy as np

def construct_packing():
    n = 26

    def layout_rows():
        rows = [6,5,6,5,4]
        max_k = max(rows)
        margin = 1e-6
        s = (1.0-2*margin)/(max_k-1) if max_k>1 else 0.5
        h = s*np.sqrt(3)/2.0
        R = len(rows)
        span_v = (R-1)*h
        if span_v > 1.0-2*margin:
            h = (1.0-2*margin)/(R-1); s = h*2.0/np.sqrt(3)
        y0 = 0.5 - ((R-1)*h)/2.0
        centers=[]
        for i,k in enumerate(rows):
            y = y0 + i*h
            left = 0.5 - ((k-1)*s)/2.0
            offset = 0.018*(0.5 - abs(y-0.5))  # small inward pull
            xs = left + np.arange(k)*s
            xs = xs*(1-offset) + 0.5*offset
            for x in xs: centers.append([x,y])
        return np.array(centers[:n])

    def layout_rings():
        cx,cy = 0.5,0.5
        centers = [[cx,cy]]
        r1, r2, r3 = 0.095, 0.195, 0.335
        for i in range(6):
            a = i*2*np.pi/6
            centers.append([cx + r1*np.cos(a), cy + r1*np.sin(a)])
        for i in range(12):
            a = i*2*np.pi/12 + np.pi/12
            centers.append([cx + r2*np.cos(a), cy + r2*np.sin(a)])
        for i in range(7):
            a = i*2*np.pi/7 + 0.07*i
            bias = 1.0 + 0.045*np.cos(2*a)
            centers.append([cx + r3*bias*np.cos(a), cy + r3*bias*np.sin(a)])
        return np.array(centers[:n])

    def layout_staggered():
        rows = [7,6,7,6]
        max_k = max(rows)
        margin = 1e-6
        s = (1.0-2*margin)/max_k
        h = s*0.95*np.sqrt(3)/2.0
        R = len(rows)
        span_v = (R-1)*h
        if span_v > 1.0-2*margin:
            h = (1.0-2*margin)/(R-1); s = h*2.0/np.sqrt(3)/0.95
        y0 = 0.5 - ((R-1)*h)/2.0
        centers=[]
        for i,k in enumerate(rows):
            y = y0 + i*h
            off = (i%2)*(s/2.0)
            left = 0.5 - ((k-1)*s)/2.0
            xs = left + np.arange(k)*s + off
            xs = np.clip(xs,0.02,0.98)
            for x in xs: centers.append([x,y])
        return np.array(centers[:n])

    candidates = [layout_rows(), layout_rings(), layout_staggered()]

    best_centers = None
    best_radii = None
    best_sum = -1.0
    for c in candidates:
        r = compute_max_radii(c)
        s = float(np.sum(r))
        if s > best_sum:
            best_sum = s; best_centers = c.copy(); best_radii = r.copy()

    # Short deterministic greedy center-adjustment pass:
    # Try small moves on each center (grid of offsets) and accept only if sum improves.
    # This is a local deterministic refinement (no random).
    deltas = [0.0,  0.02, -0.02, 0.01, -0.01]
    # try coarse then fine pass
    for radius_step in [0,1]:
        improved = False
        for i in range(best_centers.shape[0]):
            base = best_centers.copy()
            current_sum = float(np.sum(best_radii))
            best_local_sum = current_sum
            best_local_pos = best_centers[i].copy()
            for dx in deltas:
                for dy in deltas:
                    if dx==0.0 and dy==0.0 and radius_step==1:
                        continue
                    cand = base.copy()
                    cand[i,0] = np.clip(cand[i,0] + dx, 0.0, 1.0)
                    cand[i,1] = np.clip(cand[i,1] + dy, 0.0, 1.0)
                    r_cand = compute_max_radii(cand, max_iters=400, tol=1e-7)
                    s_cand = float(np.sum(r_cand))
                    if s_cand > best_local_sum + 1e-9:
                        best_local_sum = s_cand
                        best_local_pos = cand[i].copy()
                        best_local_radii = r_cand
            if best_local_sum > current_sum + 1e-9:
                best_centers[i] = best_local_pos
                best_radii = best_local_radii
                best_sum = best_local_sum
                improved = True
        if not improved:
            break

    return best_centers, best_radii, float(np.sum(best_radii))

def compute_max_radii(centers, max_iters=800, tol=1e-8):
    """
    Deterministic radius relaxation:
     - start with wall-limited radii
     - iteratively shrink to resolve overlaps using per-pair scaling factors
     - re-apply wall limits and grow cautiously toward allowed slack
    """
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    if n == 0:
        return np.array([])
    # initial wall limits
    border = np.minimum.reduce([centers[:,0], centers[:,1], 1.0-centers[:,0], 1.0-centers[:,1]])
    radii = np.maximum(border.copy(), 0.0)

    if n==1:
        return radii

    diffs = centers[:,None,:] - centers[None,:,:]
    D = np.hypot(diffs[...,0], diffs[...,1])
    np.fill_diagonal(D, np.inf)

    for it in range(max_iters):
        old = radii.copy()

        ri = radii[:,None]; rj = radii[None,:]
        sumr = ri + rj
        viol = sumr > D
        if np.any(viol):
            factors = np.ones_like(sumr)
            factors[viol] = D[viol] / sumr[viol]
            # for each circle, most restrictive factor across pairs
            min_scale = factors.min(axis=1)
            min_scale = np.maximum(min_scale, 1e-12)
            radii *= min_scale

        # enforce wall limits
        radii = np.minimum(radii, border)

        # compute allowed growth: allowed_i = min(border_i, min_j D_ij - r_j)
        slack = D - radii[None,:]
        allowed = np.minimum(border, slack.min(axis=1))
        allowed = np.maximum(allowed, 0.0)
        grow_mask = allowed > radii
        # grow fractionally to avoid oscillation
        radii[grow_mask] = radii[grow_mask] + 0.5*(allowed[grow_mask] - radii[grow_mask])

        if np.max(np.abs(radii-old)) < tol:
            break

    return np.maximum(radii, 0.0)

# EVOLVE-BLOCK-END

# fixed runtime interface
def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

def visualize(centers, radii):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
    for c,r in zip(centers, radii):
        ax.add_patch(Circle(c, r, alpha=0.6, ec='k'))
    plt.show()

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s:.6f}")