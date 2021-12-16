import importlib
import sys
import traceback


def run_experiment(dataset, dataset_size, num_classes, net, ssl_method, pretraining, amount_labelled):
    sys_main = importlib.import_module(ssl_method).sys_main

    save_name = f"{dataset}_{net}_{ssl_method}_{pretraining}_{amount_labelled}"

    epoch_size_in_iterations = amount_labelled // 64 + 1

    # Args are supplied args + common settings from config + custom
    args = list(sys.argv[1:])
    args += [
        "--c", f"config/experiments/{ssl_method}.yaml",
        "--save_name", save_name,
        "--net", net,
        "--pretrained", str(pretraining),
        "--dataset", dataset,
        "--retain-epochs", "50", "100", "500", "1000",
        "--num_labels", str(amount_labelled),
        "--num_train_iter", str(1000 * epoch_size_in_iterations + 1),
        "--num_eval_iter", str(epoch_size_in_iterations),
        "--num_classes", str(num_classes)
    ]

    try:
        sys_main(args)
    except Exception:
        with open(f"{save_name}.err", "w") as file:
            file.write(traceback.format_exc())


if __name__ == '__main__':
    IMAGENET_PRETRAINING_SETTING = (False, True)
    NET_SETTING = ("efficientnet_b0", "efficientnet_b2", "efficientnet_b4", "efficientnet_b6")
    SSL_METHOD_SETTING = ("uda", "fixmatch", "flexmatch", "fullysupervised")
    DATASETS_SETTING = {  # (dataset_size, num_classes, (...amount_labelled))
        "cifar10": (50000, 10, (40, 250, 4000)),
        "cifar100": (50000, 100, (400, 2500, 10000)),
        "stl10": (5000, 10, (40, 250, 1000)),
        "svhn": (73257 + 531131, 10, (40, 250, 1000)),
        "custom_birds_0.0_1.0": (10061, 200, (1000, 3000, 8000))
    }

    for dataset in DATASETS_SETTING.keys():
        dataset_size, num_classes, amount_labelled_setting = DATASETS_SETTING[dataset]
        for net in NET_SETTING:
            for ssl_method in SSL_METHOD_SETTING:
                for pretraining in IMAGENET_PRETRAINING_SETTING:
                    # Run the fully-supervised method on the entire dataset as well
                    if ssl_method == "fullysupervised":
                        run_experiment(
                            dataset,
                            dataset_size,
                            num_classes,
                            net,
                            ssl_method,
                            pretraining,
                            -1
                        )

                    for amount_labelled in amount_labelled_setting:
                        run_experiment(
                            dataset,
                            dataset_size,
                            num_classes,
                            net,
                            ssl_method,
                            pretraining,
                            amount_labelled
                        )
