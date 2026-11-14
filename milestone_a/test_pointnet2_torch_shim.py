"""Checks the PyTorch shim against naive loop references of the CUDA semantics."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import pointnet2_torch_shim as shim  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def naive_fps(xyz, npoint):
    B, N, _ = xyz.shape
    out = torch.zeros(B, npoint, dtype=torch.long)
    for b in range(B):
        temp = torch.full((N,), 1e10)
        far = 0
        for i in range(npoint):
            out[b, i] = far
            d = ((xyz[b] - xyz[b, far]) ** 2).sum(-1)
            temp = torch.minimum(temp, d)
            far = int(temp.argmax())
    return out


def naive_ball_query(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    P = new_xyz.shape[1]
    out = torch.zeros(B, P, nsample, dtype=torch.long)
    for b in range(B):
        for p in range(P):
            hits = []
            for k in range(N):
                d2 = ((new_xyz[b, p] - xyz[b, k]) ** 2).sum()
                if d2 < radius * radius:
                    hits.append(k)
                    if len(hits) == nsample:
                        break
            if hits:
                row = hits + [hits[0]] * (nsample - len(hits))
                out[b, p] = torch.tensor(row)
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_fps_matches_naive(device):
    torch.manual_seed(0)
    xyz = torch.rand(2, 200, 3)
    out = torch.zeros(2, 32, dtype=torch.int32, device=device)
    temp = torch.zeros(2, 200, device=device)
    shim.furthest_point_sampling_wrapper(2, 200, 32, xyz.to(device), temp, out)
    assert torch.equal(out.cpu().long(), naive_fps(xyz, 32))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("radius", [0.05, 0.3, 5.0])
def test_ball_query_matches_naive_including_padding_and_empty(device, radius):
    torch.manual_seed(1)
    xyz = torch.rand(2, 120, 3)
    new_xyz = torch.cat([xyz[:, :10], torch.full((2, 2, 3), 50.0)], dim=1)  # last two have no neighbours
    idx = torch.zeros(2, 12, 16, dtype=torch.int32, device=device)
    shim.ball_query_wrapper(2, 120, 12, radius, 16, new_xyz.to(device), xyz.to(device), idx)
    assert torch.equal(idx.cpu().long(), naive_ball_query(radius, 16, xyz, new_xyz))


@pytest.mark.parametrize("device", DEVICES)
def test_ball_query_chunking_is_consistent(device, monkeypatch):
    torch.manual_seed(2)
    xyz = torch.rand(2, 300, 3, device=device)
    new_xyz = xyz[:, :100].contiguous()
    whole = torch.zeros(2, 100, 8, dtype=torch.int32, device=device)
    shim.ball_query_wrapper(2, 300, 100, 0.2, 8, new_xyz, xyz, whole)
    monkeypatch.setattr(shim, "_CHUNK_ELEMENTS", 2 * 300 * 7)  # forces 7-row chunks
    chunked = torch.zeros_like(whole)
    shim.ball_query_wrapper(2, 300, 100, 0.2, 8, new_xyz, xyz, chunked)
    assert torch.equal(whole, chunked)


@pytest.mark.parametrize("device", DEVICES)
def test_three_nn_and_interpolate(device):
    torch.manual_seed(3)
    unknown = torch.rand(2, 50, 3, device=device)
    known = torch.rand(2, 20, 3, device=device)
    dist2 = torch.zeros(2, 50, 3, device=device)
    idx = torch.zeros(2, 50, 3, dtype=torch.int32, device=device)
    shim.three_nn_wrapper(2, 50, 20, unknown, known, dist2, idx)
    full = torch.cdist(unknown, known).square()
    ref_d, ref_i = torch.sort(full, dim=-1)
    assert torch.allclose(dist2, ref_d[..., :3], atol=1e-6)
    assert torch.equal(idx.long(), ref_i[..., :3])
    assert bool((dist2[..., 0] <= dist2[..., 1]).all() and (dist2[..., 1] <= dist2[..., 2]).all())

    feats = torch.rand(2, 5, 20, device=device)
    weight = torch.rand(2, 50, 3, device=device)
    out = torch.zeros(2, 5, 50, device=device)
    shim.three_interpolate_wrapper(2, 5, 20, 50, feats, idx, weight, out)
    ref = torch.zeros_like(out)
    for b in range(2):
        for n in range(50):
            for k in range(3):
                ref[b, :, n] += feats[b, :, idx[b, n, k].long()] * weight[b, n, k]
    assert torch.allclose(out, ref, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_gather_group_and_gradients(device):
    torch.manual_seed(4)
    feats = torch.rand(2, 4, 30, device=device, requires_grad=False)
    idx = torch.randint(0, 30, (2, 7), device=device, dtype=torch.int32)
    out = torch.zeros(2, 4, 7, device=device)
    shim.gather_points_wrapper(2, 4, 30, 7, feats, idx, out)
    assert torch.equal(out, feats[:, :, :].gather(2, idx.long().unsqueeze(1).expand(2, 4, 7)))
    grad_out = torch.rand(2, 4, 7, device=device)
    grad = torch.zeros(2, 4, 30, device=device)
    shim.gather_points_grad_wrapper(2, 4, 30, 7, grad_out, idx, grad)
    leaf = feats.clone().requires_grad_(True)
    leaf.gather(2, idx.long().unsqueeze(1).expand(2, 4, 7)).backward(grad_out)
    assert torch.allclose(grad, leaf.grad, atol=1e-6)

    gidx = torch.randint(0, 30, (2, 6, 5), device=device, dtype=torch.int32)
    gout = torch.zeros(2, 4, 6, 5, device=device)
    shim.group_points_wrapper(2, 4, 30, 6, 5, feats, gidx, gout)
    for b in range(2):
        for p in range(6):
            for s in range(5):
                assert torch.equal(gout[b, :, p, s], feats[b, :, gidx[b, p, s].long()])
    ggrad_out = torch.rand(2, 4, 6, 5, device=device)
    ggrad = torch.zeros(2, 4, 30, device=device)
    shim.group_points_grad_wrapper(2, 4, 30, 6, 5, ggrad_out, gidx, ggrad)
    leaf2 = feats.clone().requires_grad_(True)
    flat = gidx.long().reshape(2, 1, 30).expand(2, 4, 30)
    leaf2.gather(2, flat).reshape(2, 4, 6, 5).backward(ggrad_out)
    assert torch.allclose(ggrad, leaf2.grad, atol=1e-6)

    tidx = torch.randint(0, 30, (2, 9, 3), device=device, dtype=torch.int32)
    tw = torch.rand(2, 9, 3, device=device)
    tgo = torch.rand(2, 4, 9, device=device)
    tgrad = torch.zeros(2, 4, 30, device=device)
    shim.three_interpolate_grad_wrapper(2, 4, 9, 30, tgo, tidx, tw, tgrad)
    leaf3 = feats.clone().requires_grad_(True)
    picked = leaf3.gather(2, tidx.long().reshape(2, 1, 27).expand(2, 4, 27)).reshape(2, 4, 9, 3)
    (picked * tw.unsqueeze(1)).sum(-1).backward(tgo)
    assert torch.allclose(tgrad, leaf3.grad, atol=1e-6)
