from typing import Literal, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def select_tokens(
    obj_masks: torch.Tensor,                 # (O, N, H_rz, W_rz) or (N, H_rz, W_rz)
    grid_thw: Tuple[int, int, int],          # (T, H, W) patch grid *after* temporal merging, *before* spatial merging
    *,
    patch_size: int = 14,
    spatial_merge_size: int = 2,             # m
    temporal_patch_size: int = 2,            # g
    coverage_thresh: float = 0.5,
    time_reduce: Literal["mean", "max", "min"] = "max",  # reduction across the g frames merged into one grid
    device: str | torch.device = "cpu",
    retry_step: float = 0.2,
    retry_times: int = 2,
    ensure_at_least_one: bool = True,
    dtype: torch.dtype = torch.float32,
):
    """Select the visual tokens covered by each object's segmentation masks.

    Returns:
        union_idx:      (K,)          selected token indices, union over all objects
        per_obj_idx:    List[(Ki,)]   selected token indices for each object
        per_obj_cover:  (O, T, Hm, Wm) coverage map for each object

    Constraints: N == T * g, and H_rz / W_rz must be divisible by (m * patch_size)
    (guaranteed by the video preprocessing).
    """
    if obj_masks.dim() == 3:
        obj_masks = obj_masks.unsqueeze(0)
    O, N, H_rz, W_rz = obj_masks.shape
    T, H, W = grid_thw
    m, g = spatial_merge_size, temporal_patch_size
    if N != T * g:
        if N < T * g:
            pad = T * g - N
            last = obj_masks[:, -1:, :, :].repeat(1, pad, 1, 1)
            obj_masks = torch.cat([obj_masks, last], dim=1)
            N = T * g
        else:
            obj_masks = obj_masks[:, :T * g, :, :]
            N = T * g
    Hm, Wm = H // m, W // m
    pix_h, pix_w = m * patch_size, m * patch_size
    assert H_rz % pix_h == 0 and W_rz % pix_w == 0, "resized height/width must be divisible by spatial_merge_size * patch_size"

    M = obj_masks.to(device=device, dtype=dtype).clamp(0, 1)

    # Per-grid coverage: fraction of each (m*patch_size)^2 cell covered by the mask.
    M_flat = M.view(O * N, 1, H_rz, W_rz)
    cov_hw = F.avg_pool2d(M_flat, kernel_size=(pix_h, pix_w), stride=(pix_h, pix_w))
    cov_hw = cov_hw.view(O, N, Hm, Wm)

    cov_hw = cov_hw.view(O, T, g, Hm, Wm)
    if time_reduce == "mean":
        cov_thw = cov_hw.mean(dim=2)
    elif time_reduce == "max":
        cov_thw = cov_hw.max(dim=2).values
    elif time_reduce == "min":
        cov_thw = cov_hw.min(dim=2).values
    else:
        raise ValueError("time_reduce must be one of {'mean', 'max', 'min'}")

    per_obj_idx = []
    per_t = Hm * Wm
    for o in range(O):
        # Threshold the coverage map; if nothing passes, relax the threshold a few times.
        nz = torch.empty(0, 3, dtype=torch.long, device=device)
        tried = 0
        thr = coverage_thresh
        while tried <= retry_times:
            thr_eff = max(0.0, float(thr))
            sel = (cov_thw[o] >= thr_eff)
            nz = torch.nonzero(sel, as_tuple=False)
            if nz.numel() > 0:
                break
            tried += 1
            thr -= retry_step
        if nz.numel() == 0:
            if ensure_at_least_one:
                # Fall back to the single best-covered token.
                flat = cov_thw[o].reshape(-1)
                arg = torch.argmax(flat)
                t = arg // (Hm * Wm)
                rem = arg % (Hm * Wm)
                hp = rem // Wm
                wp = rem % Wm
                idx = (t * per_t + hp * Wm + wp).view(1)
                per_obj_idx.append(idx.to(device=device, dtype=torch.long))
            else:
                per_obj_idx.append(torch.empty(0, dtype=torch.long, device=device))
        else:
            t = nz[:, 0]
            hp = nz[:, 1]
            wp = nz[:, 2]
            idx = t * per_t + hp * Wm + wp
            per_obj_idx.append(idx.to(device=device, dtype=torch.long))

    if len(per_obj_idx) == 0:
        union_idx = torch.empty(0, dtype=torch.long, device=device)
    else:
        union_idx = torch.unique(torch.cat(per_obj_idx, dim=0)) if per_obj_idx[0].numel() else torch.empty(0, dtype=torch.long, device=device)

    union_idx_cpu = union_idx.cpu()
    per_obj_idx_cpu = [idx.cpu() for idx in per_obj_idx]
    cov_thw_cpu = cov_thw.cpu()

    del M, M_flat, cov_hw, cov_thw, per_obj_idx, union_idx
    if O > 0:
        del sel, nz

    return union_idx_cpu, per_obj_idx_cpu, cov_thw_cpu
