import argparse
import math
import traceback

from diffeo_offline_analysis import load_model_and_eval_loader
from diffeo.eval import calculate_diffeo_d_g

from sklearn.metrics import accuracy_score
import torch


def calc_top1(model, eval_loader, gpu):
    y_true = []
    y_pred = []
    for _, x, y in eval_loader:
        x, y = x.cuda(gpu), y.cuda(gpu)
        logits = model(x)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
    return accuracy_score(y_true, y_pred)


def stats(d, g, top1):
    e = 1.0 - top1
    try:
        r = d / g
    except ZeroDivisionError:
        r = 0.0

    r_coeff = e / math.sqrt(r)

    return e, r, r_coeff


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="diffeo_all",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest='gpu',
        metavar='GPU',
        type=int,
        help='The GPU to use'
    )

    parsed = parser.parse_args(args=args)

    IMAGENET_PRETRAINING_SETTING = (False, True)
    NET_SETTING = ("efficientnet_b0", "efficientnet_b2", "efficientnet_b4", "efficientnet_b6")
    SSL_METHOD_SETTING = ("uda", "fixmatch", "flexmatch", "fullysupervised")
    DATASETS_SETTING = {  # (dataset_size, num_classes, (...amount_labelled))
        "cifar10": (50000, 10, (40, 250, 4000)),
        "cifar100": (50000, 100, (400, 2500, 10000)),
        "stl10": (105000, 10, (40, 250, 1000)),
        "svhn": (73257 + 531131, 10, (40, 250, 1000)),
        "custom_birds_0.0_1.0": (10061, 200, (1000, 3000, 8000))
    }
    EPOCHS = (50, 100, 500, 1000)
    print("dataset,net,ssl_method,pretraining,amount_labelled,epoch,error,D,G,R,R_coeff")

    for dataset in DATASETS_SETTING.keys():
        dataset_size, num_classes, amount_labelled_setting = DATASETS_SETTING[dataset]
        for net in NET_SETTING:
            for ssl_method in SSL_METHOD_SETTING:
                for pretraining in IMAGENET_PRETRAINING_SETTING:
                    for epoch in EPOCHS:
                        # Run the fully-supervised method on the entire dataset as well
                        if ssl_method == "fullysupervised":
                            try:
                                model, loader_dict = load_model_and_eval_loader(
                                    dataset,
                                    num_classes,
                                    net,
                                    ssl_method,
                                    pretraining,
                                    -1,
                                    epoch,
                                    parsed.gpu
                                )
                                top1 = calc_top1(model.model, loader_dict['eval'], parsed.gpu)
                                d, g = calculate_diffeo_d_g(model.model, loader_dict['eval'], parsed.gpu)
                                e, r, r_coeff = stats(d, g, top1)
                            except Exception as e:
                                e = d = g = r = ""
                                r_coeff = str(e)
                            print(f"{dataset},{net},{ssl_method},{pretraining},-1,{epoch},{e},{d},{g},{r},{r_coeff}")

                        for amount_labelled in amount_labelled_setting:
                            try:
                                model, loader_dict = load_model_and_eval_loader(
                                    dataset,
                                    num_classes,
                                    net,
                                    ssl_method,
                                    pretraining,
                                    amount_labelled,
                                    epoch,
                                    parsed.gpu
                                )
                                top1 = calc_top1(model.model, loader_dict['eval'], parsed.gpu)
                                d, g = calculate_diffeo_d_g(model.model, loader_dict['eval'], parsed.gpu)
                                e, r, r_coeff = stats(d, g, top1)
                            except Exception as e:
                                e = d = g = r = ""
                                r_coeff = str(e)
                            print(f"{dataset},{net},{ssl_method},{pretraining},{amount_labelled},{epoch},{e},{d},{g},{r},{r_coeff}")


def sys_main() -> int:
    try:
        main()
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys_main()