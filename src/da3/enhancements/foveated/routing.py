"""Mixed per-view resolutions in ONE DA3 forward pass (enhancement #2, Foveated DA3).

docs/da3_enhancement_report.md §4.3. Every view keeps its own resolution:

* patch embedding, the learned position embedding (interpolated to each grid), the within-view
  blocks with their 2D RoPE grid, the Dual-DPT head and the camera decoding run once per
  resolution group;
* the cross-view blocks run on the concatenation of all views' tokens. This is well defined because
  the released backbone gives every patch token the same constant RoPE position in those blocks
  (`MultiViewViT.rope_positions`), so cross-view attention depends neither on where a token sits
  in its image nor on how many tokens a view has;
* the camera head D_C reads one camera token per view: poses and FOVs (angles) do not depend on
  the resolution, intrinsics are decoded in each view's own pixels.

`foveated_forward` mirrors `DA3.forward` / `MultiViewViT.forward` operation by operation (reference
view selection and reordering, camera tokens, RoPE, concatenated head features, chunked Dual-DPT),
so with all views at one resolution the outputs are bit-identical to the original
(tests/test_foveated.py on CPU; `scripts/foveated_eval.py parity` on the GPU with the released
weights). The network and its weights are never changed.

Attention-mass balancing (`balance=True`, optional, training-free). DA3 was trained with every view
contributing the same number of keys to cross-view attention; a 252-px context view contributes
about 4x fewer than a 504-px focus view, so it gets less attention mass. Adding log(N_ref / N_v) to
the logits of view v's patch keys (N = patch tokens, N_ref = largest view) gives exactly the
softmax that replicating each of its patch tokens N_ref / N_v times would give, i.e. the attention
sees the context view as if its token grid were upsampled to the focus resolution, at no extra
attention cost. Within-view attention is unchanged. Implemented without an (N_q x N_k) mask: one
extra channel carries the bias (`sdpa_key_bias`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from da3.backbone.vit import (
    REF_SELECTION_MIN_VIEWS,
    from_within_view,
    select_reference_view,
    to_cross_view,
    to_within_view,
)
from da3.camera import affine_inverse, decode_pose, mat_to_quat, normalize_extrinsics
from da3.geometry import pose_from_ray_map

OUTPUT_KEYS = ("depth", "depth_conf", "ray", "ray_conf")


@dataclass
class Groups:
    """Views grouped by resolution, in order of first appearance."""

    hw: list        # [(H, W)] per group
    members: list   # [[original view indices, ascending]] per group

    def __len__(self) -> int:
        return len(self.hw)


def group_views(hws, group_ids=None) -> Groups:
    """Group views of sizes `hws` by (H, W). `group_ids` (one label per view) splits groups further
    (used by the tests to route same-size views through separate groups)."""
    keys = [tuple(hw) for hw in hws] if group_ids is None else [(tuple(hw), g) for hw, g in zip(hws, group_ids)]
    order, members = [], {}
    for s, key in enumerate(keys):
        if key not in members:
            members[key] = []
            order.append(key)
        members[key].append(s)
    return Groups(hw=[tuple(hws[members[k][0]]) for k in order], members=[members[k] for k in order])


# ----------------------------------------------------------------------------- balanced attention
def sdpa_key_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
    """softmax(q k^T / sqrt(d) + key_bias) v for q, k, v (B, heads, N, d) and key_bias (N_k,) added to
    every query's logits. No (N_q, N_k) mask is built: q -> [q, 1, 0...], k -> [k, b sqrt(d), 0...],
    v -> [v, 0...] (channels padded to d + 8 so the fused kernels still apply), explicit scale."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    pad = 8
    zq = q.new_zeros(*q.shape[:-1], pad)
    zq[..., 0] = 1.0
    zk = k.new_zeros(*k.shape[:-1], pad)
    zk[..., 0] = (key_bias.float() / scale).to(k.dtype)
    zv = v.new_zeros(*v.shape[:-1], pad)
    out = F.scaled_dot_product_attention(torch.cat([q, zq], -1), torch.cat([k, zk], -1), torch.cat([v, zv], -1),
                                         scale=scale)
    return out[..., :d]


def _attention_key_bias(attn, x, pos, key_bias):
    """`backbone.layers.Attention.forward` with `sdpa_key_bias` in place of plain SDPA."""
    B, N, C = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rope is not None and pos is not None:
        q, k = attn.rope(q, pos), attn.rope(k, pos)
    if q.dtype != v.dtype:
        q, k = q.to(v.dtype), k.to(v.dtype)
    x = sdpa_key_bias(q, k, v, key_bias)
    return attn.proj(x.transpose(1, 2).reshape(B, N, C))


def _block_key_bias(blk, x, pos, key_bias):
    """`backbone.layers.Block.forward` with key-biased attention."""
    x = x + blk.ls1(_attention_key_bias(blk.attn, blk.norm1(x), pos, key_bias))
    return x + blk.ls2(blk.mlp(blk.norm2(x)))


def mass_balance_bias(token_counts, n_views, n_special: int, device) -> torch.Tensor:
    """Key bias for a cross-view sequence made of groups with `token_counts[g]` tokens per view
    (special tokens included) and `n_views[g]` views: log(N_ref / N_g) on patch tokens, 0 on special
    tokens, N = patch tokens per view, N_ref = the largest."""
    patches = [c - n_special for c in token_counts]
    ref = max(patches)
    parts = []
    for c, p, s in zip(token_counts, patches, n_views):
        b = torch.zeros(c, device=device)
        b[n_special:] = math.log(ref / p)
        parts.append(b.repeat(s))
    return torch.cat(parts)


# ----------------------------------------------------------------------------- camera tokens (E_c)
def camera_tokens(model, w2c: torch.Tensor, K: torch.Tensor, hws) -> torch.Tensor:
    """E_c tokens (1, S, D) for views of different resolutions. w2c (S, 3|4, 4) world-to-camera;
    K (S, 3, 3) in pixels of each view's own (H, W). The FOV in the 9-D encoding is an angle, so the
    same camera gives the same token at any resolution. Same operations as `CameraEncoder.forward`
    (a Python-scalar (H / 2) / f is evaluated by PyTorch as f.reciprocal() * (H / 2))."""
    enc_model = model.cam_enc
    if enc_model is None:
        raise ValueError("model has no camera encoder")
    dev = K.device
    with torch.autocast(dev.type, enabled=False):
        w2c_n = normalize_extrinsics(w2c.float()[None])          # relative to view 0, median-distance scaled
        c2w = affine_inverse(w2c_n)
        Kf = K.float()[None]
        H = torch.tensor([float(hw[0]) for hw in hws], device=dev)
        W = torch.tensor([float(hw[1]) for hw in hws], device=dev)
        fov_h = 2 * torch.atan(Kf[..., 1, 1].reciprocal() * (H / 2))
        fov_w = 2 * torch.atan(Kf[..., 0, 0].reciprocal() * (W / 2))
        q = mat_to_quat(c2w[..., :3, :3])
        enc = torch.cat([c2w[..., :3, 3], q, fov_h[..., None], fov_w[..., None]], -1).float()
        x = enc_model.pose_branch(enc)
        if enc_model.trunk is not None:
            x = enc_model.trunk_norm(enc_model.trunk(enc_model.token_norm(x)))
    return x


def _stack_views(views) -> torch.Tensor:
    """Stack (3, H, W) views into a batch with their memory layout: views cut from a channels-last
    batch (io_utils.to_model_input permutes HWC -> CHW) stay channels-last, so the patch-embedding
    convolution runs the same kernel as on the original batch (bit-identical outputs)."""
    x = torch.stack(list(views))
    if all(v.dim() == 3 and v.shape[0] > 1 and v.stride(0) == 1 for v in views):
        x = x.contiguous(memory_format=torch.channels_last)
    return x


@dataclass
class Prepared:
    """Views already stacked per resolution group and moved to the device: (1, S_g, 3, H_g, W_g)."""

    hws: list
    groups: Groups
    batches: list


def prepare_views(views, device, group_ids=None) -> Prepared:
    """Stack the views of every resolution group and move them to `device` (lets callers time the
    forward pass alone and keep a single copy of the input on the GPU; foveated_forward releases each
    batch right after its patch embedding)."""
    hws = [tuple(int(t) for t in v.shape[-2:]) for v in views]
    groups = group_views(hws, group_ids)
    batches = [_stack_views([views[s] for s in m]).to(device)[None] for m in groups.members]
    return Prepared(hws=hws, groups=groups, batches=batches)


def _learnable_tokens(bb, S: int, ref_view: int) -> torch.Tensor:
    """(1, S, D) learnable camera tokens in original view order: the reference-view token for
    `ref_view`, the shared token for all others (one shared token if the model has only one)."""
    tok = bb.camera_token
    if tok.shape[1] == 1:
        return tok.expand(1, S, -1)
    out = tok[:, 1:2].expand(1, S, -1).clone()
    out[:, ref_view] = tok[:, 0]
    return out


def _gather_cls(xs, order, S: int) -> torch.Tensor:
    """(1, S, 1, D): token 0 of every view in original order (what select_reference_view reads)."""
    cls = xs[0].new_empty(1, S, 1, xs[0].shape[-1])
    for g, x in enumerate(xs):
        cls[:, order[g], 0] = x[:, :, 0]
    return cls


# ----------------------------------------------------------------------------- forward
@torch.no_grad()
def foveated_forward(model, views, w2c: torch.Tensor | None = None, K: torch.Tensor | None = None,
                     ref_strategy: str = "first", ref_candidates=None, balance: bool = False, group_ids=None,
                     keep=OUTPUT_KEYS, offload: bool = False) -> dict:
    """One DA3 forward pass over the views of ONE scene, each at its own resolution.

    views: S tensors (3, H_s, W_s), ImageNet-normalised, sides multiples of the patch size; or the
      `Prepared` result of prepare_views (its batches are released after the patch embedding).
    w2c (S, 3|4, 4), K (S, 3, 3): optional known cameras (K in pixels of each view's own size) -> E_c.
    ref_strategy: as DA3.forward; ref_candidates: restrict the released reference-view selection to
      these views (e.g. the focus views). With "first" the reference is the first candidate.
    balance: attention-mass balancing in the cross-view blocks (module docstring).
    keep: output maps to keep; offload: move them to the CPU chunk by chunk (large view counts).
    Returns {key: [S maps]} for the kept maps (depth / depth_conf (H_s, W_s), ray (h_s, w_s, 6),
    ray_conf (h_s, w_s)), "pose_enc" (S, 9), "ref_index" (int | None), "hw" [(H_s, W_s)], "groups".
    Call it under torch.autocast like DA3.forward (the head, D_C and E_c run in fp32 regardless).
    """
    bb = model.backbone
    dev = next(model.parameters()).device
    prep = views if isinstance(views, Prepared) else prepare_views(views, dev, group_ids)
    del views
    hws, groups = prep.hws, prep.groups
    S, G = len(hws), len(groups)
    if S == 0:
        raise ValueError("no views")
    p, D = bb.patch_size, bb.embed_dim
    for hw in groups.hw:
        if hw[0] % p or hw[1] % p:
            raise ValueError(f"view size {hw} is not a multiple of the patch size {p}")

    # ---- embeddings, one group at a time (as MultiViewViT.forward does for all views)
    xs, local_pos, const_pos = [], [], []
    for g in range(G):
        img = prep.batches[g].to(dev)
        prep.batches[g] = None
        _, Sg, _, H, W = img.shape
        gh, gw = H // p, W // p
        x = bb.patch_embed(img.flatten(0, 1))
        x = torch.cat([bb.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = (x + bb.interpolate_pos_embed(gh, gw)).reshape(1, Sg, gh * gw + 1, -1)
        xs.append(x)
        if bb.rope is not None:
            lp, cp = bb.rope_positions(1, Sg, gh, gw, x.device)
            local_pos.append(lp)
            const_pos.append(cp)
        del img

    posed = w2c is not None
    cam_tok = None
    if posed:
        if K is None:
            raise ValueError("K is required together with w2c")
        cam_tok = camera_tokens(model, w2c.to(dev), K.to(dev), hws)

    order = [list(m) for m in groups.members]   # current view order inside each group
    seq = list(range(G))                         # group order in the cross-view sequence
    ref = None
    if bb.cam_token_mode == "prepend":           # paper presets: tokens by original index (view 0 = reference)
        tok = cam_tok if posed else _learnable_tokens(bb, S, 0)
        for g in range(G):
            xs[g] = torch.cat([tok[:, order[g], None].to(xs[g].dtype), xs[g]], dim=2)

    local_xs = list(xs)
    feats, cam_feats = [], []
    for i, blk in enumerate(bb.blocks):
        if bb.ref_view_selection and i == bb.alt_start - 1 and S >= REF_SELECTION_MIN_VIEWS and not posed:
            cand = list(range(S)) if ref_candidates is None else sorted(int(c) for c in ref_candidates)
            cls = _gather_cls(xs, order, S)
            ref = cand[int(select_reference_view(cls[:, cand], ref_strategy)[0])]
            g_ref = next(g for g in range(G) if ref in order[g])
            j = order[g_ref].index(ref)
            perm = [j] + [t for t in range(len(order[g_ref])) if t != j]   # as reference_permutation
            xs[g_ref], local_xs[g_ref] = xs[g_ref][:, perm], local_xs[g_ref][:, perm]
            order[g_ref] = [order[g_ref][t] for t in perm]
            seq = [g_ref] + [g for g in seq if g != g_ref]
        if bb.cam_token_mode == "replace_cls" and i == bb.alt_start:
            tok = cam_tok if posed else _learnable_tokens(bb, S, 0 if ref is None else ref)
            for g in range(G):
                xs[g] = torch.cat([tok[:, order[g], None].to(xs[g].dtype), xs[g][:, :, 1:]], dim=2)

        use_rope = bb.rope is not None and i >= bb.rope_start
        if bb.is_cross_view(i):
            parts = [to_cross_view(xs[g]) for g in seq]
            x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
            pos = None
            if use_rope:
                pp = [to_cross_view(const_pos[g]) for g in seq]
                pos = pp[0] if len(pp) == 1 else torch.cat(pp, dim=1)
            counts = [xs[g].shape[2] for g in seq]
            if balance and len(set(counts)) > 1:
                bias = mass_balance_bias(counts, [xs[g].shape[1] for g in seq], bb.n_special, x.device)
                x = _block_key_bias(blk, x, pos, bias)
            else:
                x = bb._run_block(blk, x, pos)
            sizes = [xs[g].shape[1] * xs[g].shape[2] for g in seq]
            for g, part in zip(seq, torch.split(x, sizes, dim=1)):
                xs[g] = part.reshape(1, xs[g].shape[1], xs[g].shape[2], -1)
        else:
            for g in range(G):
                pos = to_within_view(local_pos[g]) if use_rope else None
                xs[g] = from_within_view(bb._run_block(blk, to_within_view(xs[g]), pos), 1, xs[g].shape[1])
            local_xs = list(xs)

        if i in bb.out_layers:
            layer, cams = [], [None] * S
            for g in range(G):
                out = torch.cat([local_xs[g], xs[g]], dim=-1) if bb.cat_token else bb.norm(xs[g])
                if order[g] != groups.members[g]:   # back to ascending original order
                    out = out[:, [order[g].index(s) for s in groups.members[g]]]
                cam = out[:, :, 0]
                if bb.cat_token:
                    out = torch.cat([out[..., :D], bb.norm(out[..., D:])], -1)
                layer.append(out[:, :, bb.n_special:])
                for j, s in enumerate(groups.members[g]):
                    cams[s] = cam[:, j]
            feats.append(layer)
            # one group: the same (strided) tensor DA3.forward hands to D_C; else views in original order
            cam_feats.append(cam if G == 1 else torch.stack(cams, dim=1))
    del xs, local_xs

    # ---- Dual-DPT per resolution group, in the same view chunks as DualDPT.forward; D_C per view
    head = model.head
    result = {}
    with torch.autocast(dev.type, enabled=False):
        for g in range(G):
            H, W = groups.hw[g]
            flat = [layer[g][0] for layer in feats]
            Sg = flat[0].shape[0]
            cs = head.chunk_size or Sg
            for i0 in range(0, Sg, cs):
                res = head._forward_views([f[i0:i0 + cs].float() for f in flat], H, W)
                for key, val in res.items():
                    if key not in keep:
                        continue
                    val = val.cpu() if offload else val
                    store = result.setdefault(key, [None] * S)
                    for j in range(val.shape[0]):
                        store[groups.members[g][i0 + j]] = val[j]
                del res
            for layer in feats:
                layer[g] = None
        pose_enc = model.cam_head(cam_feats[-1].float())[0] if model.cam_head is not None else None
    result.update({"pose_enc": pose_enc, "ref_index": ref, "hw": hws, "groups": groups})
    return result


# ----------------------------------------------------------------------------- outputs
def cameras_from_head(pose_enc: torch.Tensor, hws) -> tuple[torch.Tensor, torch.Tensor]:
    """D_C output (S, 9) -> w2c (S, 3, 4), K (S, 3, 3) in pixels of each view's own resolution
    (decoded per resolution, exactly like DA3.cameras_from_head)."""
    S = pose_enc.shape[0]
    w2c = pose_enc.new_zeros(S, 3, 4)
    K = pose_enc.new_zeros(S, 3, 3)
    for hw in dict.fromkeys(tuple(h) for h in hws):
        idx = [s for s in range(S) if tuple(hws[s]) == hw]
        c2w, Kg = decode_pose(pose_enc[idx].float(), hw)
        w2c[idx] = affine_inverse(c2w)
        K[idx] = Kg
    return w2c, K


def cameras_from_rays(rays, hws, chunk: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray maps (list of (h_s, w_s, 6)) -> w2c (S, 3, 4), K (S, 3, 3) with our Eq. 1 + Eq. 2 solver,
    per resolution group and in chunks of views."""
    S = len(rays)
    w2c = torch.zeros(S, 3, 4, dtype=torch.float32, device=rays[0].device)
    K = torch.zeros(S, 3, 3, dtype=torch.float32, device=rays[0].device)
    for hw in dict.fromkeys(tuple(h) for h in hws):
        idx = [s for s in range(S) if tuple(hws[s]) == hw]
        for c in range(0, len(idx), chunk):
            sub = idx[c:c + chunk]
            e, k = pose_from_ray_map(torch.stack([rays[s].float() for s in sub]), hw)
            w2c[sub], K[sub] = e.float(), k.float()
    return w2c, K


def scale_K(K: torch.Tensor, src_hw, dst_hw) -> torch.Tensor:
    """Intrinsics (3, 3) for an image resized from src_hw to dst_hw (per axis)."""
    K = K.clone()
    K[0] *= dst_hw[1] / src_hw[1]
    K[1] *= dst_hw[0] / src_hw[0]
    return K


def resize_maps(maps, hw) -> torch.Tensor:
    """List of (H_s, W_s) maps -> (S, H, W), bilinear (no resampling for maps already at hw)."""
    out = []
    for m in maps:
        if tuple(m.shape[-2:]) == tuple(hw):
            out.append(m.float())
        else:
            out.append(F.interpolate(m.float()[None, None], size=tuple(hw), mode="bilinear", align_corners=False)[0, 0])
    return torch.stack(out)
