import math

import torch

from diffeo.diff import deform, temperature_range
from diffeo.image import rgb_transpose
from diffeo.typical_temperature import typical_temperature


def diffeo_imgs(imgs, cuts, interp='linear', imagenet=False, Ta=2, Tb=2, nT=30, delta=1.):
    """
    Compute noisy and diffeo versions of imgs.

    Taken from:
    https://github.com/leonardopetrini/diffeo-sota/blob/15941397685cdb1aa3ffb3ee718f5a6dde14bab3/results/utils.py#L132

    :param imgs: image samples to deform
    :param cuts: high-frequency cutoff
    :param interp: interpolation method
    :param imagenet
    :param Ta: T1 = Tmin / Ta
    :param Tb: T2 = Tmax * Tb
    :param nT: number of temperatures
    :param delta: if nT = 1, computes diffeo at delta (used for computing diffeo @ delta=1)
    :return: list of dicts, one per value of cut with deformed and noidy images
    """

    data = {}
    data['cuts'] = []

    n = imgs.shape[-2]
    if interp == 'linear':
        data['imgs'] = imgs
    else:
        if imagenet:
            data['imgs'] = torch.stack([rgb_transpose(deform(rgb_transpose(i), 0, 1, interp='gaussian')) for i in imgs])
        else:
            data['imgs'] = torch.stack([deform(i, 0, 1, interp='gaussian') for i in imgs])

    if interp == 'linear_smooth':
        imgs = data['imgs']
        interp = 'linear'

    for c in cuts:
        T1, T2 = temperature_range(n, c)
        Ts = torch.logspace(math.log10(T1 / Ta), math.log10(T2 * Tb), nT)
        if nT == 1:
            Ts = [typical_temperature(delta, c, n)]
        ds = []
        qs = []
        for T in Ts:
            # deform images
            if imagenet:
                d = torch.stack([rgb_transpose(deform(rgb_transpose(i), T, c, interp)) for i in imgs])
            else:
                d = torch.stack([deform(i, T, c, interp) for i in imgs])

            # create gaussian noise with exact same norm
            sigma = (d - data['imgs']).pow(2).sum([1, 2, 3], keepdim=True).sqrt()
            eta = torch.randn(imgs.shape)
            eta = eta / eta.pow(2).sum([1, 2, 3], keepdim=True).sqrt() * sigma
            q = data['imgs'] + eta

            ds.append(d)
            qs.append(q)

        defs = torch.stack(ds)
        nois = torch.stack(qs)

        # smoothing after adding noise
        #         if interp == 'gaussian':
        #             nois = torch.stack([deform(i, 0, 1, interp='gaussian') for i in nois])

        data['cuts'] += [{
            'cut': c,
            'temp': Ts,
            'diffeo': defs,
            'normal': nois,
        }]

    return data
