import torch


def relative_distance(f, os, ds, qs, deno=True):
    """
    Compute stabilites for a function f

    Taken from:
    https://github.com/leonardopetrini/diffeo-sota/blob/15941397685cdb1aa3ffb3ee718f5a6dde14bab3/results/utils.py#L203

    and modified to work on a single temperature.

    os: [batch, y, x, rgb]      original images
    ds: [batch, y, x, rgb]   images + gaussian noise
    qs: [batch, y, x, rgb]   diffeo images
    """
    with torch.no_grad():
        f0 = f(os).detach().reshape(len(os), -1)  # [batch, ...]
        deno = torch.cdist(f0, f0).pow(2).median().item() + 1e-10 if deno else 1
        fd = f(ds).detach().reshape(len(os), -1)  # [batch, ...]
        fq = f(qs).detach().reshape(len(os), -1)  # [batch, ...]
        outd = (fd - f0).pow(2).median(0).values.sum().item() / deno
        outq = (fq - f0).pow(2).median(0).values.sum().item() / deno
        return outd, outq
