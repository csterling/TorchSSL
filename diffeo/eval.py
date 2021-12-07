import math
from typing import Any, Dict

import torch

from .diffeo_imgs import diffeo_imgs
from .relative_distance import relative_distance


def eval_diffeo(model, eval_loader, gpu, top1) -> Dict[str, Any]:
    imgs = torch.vstack([x for _, x, _ in eval_loader]).to('cpu')

    data = diffeo_imgs(
        imgs,
        cuts=(3,),
        nT=1,
        delta=1.
    )

    d, g = relative_distance(
        model,
        imgs.to(gpu).float(),
        data['cuts'][0]['diffeo'][0].to(gpu).float(),
        data['cuts'][0]['normal'][0].to(gpu).float(),
    )

    e = 1.0 - top1
    try:
        r = d / g
    except ZeroDivisionError:
        r = 0.0

    return {
        'eval/error': e,
        'eval/D': d,
        'eval/G': g,
        'eval/R': r,
        'eval/R_coeff': e / math.sqrt(r)
    }
