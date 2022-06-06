import argparse
import importlib
import traceback

from diffeo.eval import calculate_diffeo_d_g


def diffeo_offline_analysis(
        dataset,
        num_classes,
        net,
        ssl_method,
        pretraining,
        amount_labelled,
        epoch,
        gpu
):
    save_name = f"{dataset}_{net}_{ssl_method}_{pretraining}_{amount_labelled}"

    # Load the model
    create_model = importlib.import_module(ssl_method).create_model
    parse_args = importlib.import_module(ssl_method).parse_args

    # Args are supplied args + common settings from config + custom
    args = [
        "--c", f"config/experiments/{ssl_method}.yaml",
        "--resume", "true",
        "--load_path", f"./saved_models/{save_name}/model_at_epoch_{epoch}.pth",
        "--net", net,
        "--pretrained", str(pretraining),
        "--dataset", dataset,
        "--retain-epochs", "50", "100", "500", "1000",
        "--num_labels", str(amount_labelled),
        "--num_classes", str(num_classes),
        "--multiprocessing-distributed", "false",
        "--gpu", str(gpu)
    ]

    args = parse_args(args)

    model, loader_dict, _, __ = create_model(args.gpu, 1, args)

    d, g = calculate_diffeo_d_g(model.model, loader_dict['eval'], args.gpu)

    print(
        f"D = {d}\n"
        f"G = {g}"
    )


def main(args=None):
    parser = argparse.ArgumentParser(
        prog="diffeo_offline_analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest='dataset',
        metavar='DATASET',
        choices=("cifar10", "cifar100", "stl10", "svhn", "custom_birds_0.0_1.0"),
        help='The dataset'
    )
    parser.add_argument(
        dest='num_classes',
        metavar='NUM-CLASSES',
        type=int,
        help='The number of classes in the dataset'
    )
    parser.add_argument(
        dest='net',
        metavar='NETWORK',
        choices=("efficientnet_b0", "efficientnet_b2", "efficientnet_b4", "efficientnet_b6"),
        help='The network'
    )
    parser.add_argument(
        dest='ssl_method',
        metavar='SSL-METHOD',
        choices=("uda", "fixmatch", "flexmatch", "fullysupervised"),
        help='The SSL method'
    )
    parser.add_argument(
        dest='pretraining',
        metavar='PRETRAINING',
        type=bool,
        help='Whether to use pretraining'
    )
    parser.add_argument(
        dest='amount_labelled',
        metavar='AMOUNT-LABELLED',
        type=int,
        help='Amount of labelled data'
    )
    parser.add_argument(
        dest='epoch',
        metavar='EPOCH',
        type=int,
        choices=(50, 100, 500, 1000),
        help='The epoch to evaluate'
    )
    parser.add_argument(
        dest='gpu',
        metavar='GPU',
        type=int,
        help='The GPU to use'
    )

    parsed = parser.parse_args(args=args)

    diffeo_offline_analysis(
        parsed.dataset,
        parsed.num_classes,
        parsed.net,
        parsed.ssl_method,
        parsed.pretraining,
        parsed.amount_labelled,
        parsed.epoch,
        parsed.gpu
    )


def sys_main() -> int:
    try:
        main()
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys_main()
