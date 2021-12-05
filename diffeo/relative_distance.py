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
        f0 = get_predictions(f, os).reshape(len(os), -1)  # [batch, ...]
        deno = torch.cdist(f0, f0).pow(2).median().item() + 1e-10 if deno else 1
        fd = get_predictions(f, ds).reshape(len(os), -1)  # [batch, ...]
        fq = get_predictions(f, qs).reshape(len(os), -1)  # [batch, ...]
        outd = (fd - f0).pow(2).median(0).values.sum().item() / deno
        outq = (fq - f0).pow(2).median(0).values.sum().item() / deno
        return outd, outq


def get_predictions(f, imgs):
    indv = False

    try:
        f0 = f(imgs).detach()
    except RuntimeError:
        indv = True

    if indv:
        f_os = []
        for o in imgs:
            f_o = f(o.reshape(1, *o.shape)).detach()
            f_os += [f_o]
        f0 = torch.vstack(f_os)

    return f0
