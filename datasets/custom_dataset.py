import copy
import os
from PIL import Image
from typing import Any, Dict, List, Tuple

from torchvision import transforms
from torchvision.datasets import ImageFolder

from datasets.augmentation.randaugment import RandAugment


class CustomDataset(ImageFolder):
    """
    Hook for using custom datasets.
    """
    def __init__(
        self,
        root: str,
        name: str,
        train: bool,
        mean: float,
        std: float
    ):
        super().__init__(
            root,
            transform=transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]) if train else transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        )

        dataset_list_filename = f"{name}.train" if train else f"{name}.test"

        with open(os.path.join(root, dataset_list_filename), 'r') as dataset_list_file:
            self._dataset_files = dataset_list_file.readlines()

        with open(os.path.join(root, f"{name}.labels"), 'r') as labels_file:
            self._labels = labels_file.readlines()
            self._label_indices = {
                label: index
                for index, label in enumerate(self._labels)
            }

    def _get_item_path(self, index: int) -> str:
        return os.path.join(self.root, self._dataset_files[index])

    def _get_target(self, index: int) -> int:
        target = self._label_indices[os.path.basename(os.path.dirname(self._get_item_path(index)))]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target

    def _get_image(self, index: int) -> Image.Image:
        img = Image.open(self._get_item_path(index))

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __getitem__(self, index: int) -> Any:
        return self._get_image(index), self._get_target(index)

    def __len__(self) -> int:
        return len(self.dataset_files)

    #@property
    #def data(self):
    #    images = [
    #        self._get_image(index)
    #        for index in range(len(self))
    #    ]
    #
    #    return np.array((
    #        np.array(self._get_image(index).reshape())
    #        for image in images
    #    ))


class CustomDataset2(ImageFolder):
    def __init__(self, root, name, train, ulb, mean, std):
        self.name = name
        self.train = train
        self.ulb = ulb
        super().__init__(
            root,
            transform=transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]) if train else transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        )
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if self.ulb:
            self.strong_transform = copy.deepcopy(self.transform)
            self.strong_transform.transforms.insert(0, RandAugment(3, 5))

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        with open(os.path.join(self.root, f"{self.name}.labels"), 'r') as labels_file:
            labels = [
                    label.strip()
                    for label in labels_file.readlines()
                    ]
            label_indices = {
                label: index
                for index, label in enumerate(labels)
            }
            return labels, label_indices

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_transformed = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (index, sample_transformed, target) if not self.ulb else (
            index, sample_transformed, self.strong_transform(sample))

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        dataset_list_filename = f"{self.name}.train" if self.train else f"{self.name}.test"

        with open(os.path.join(self.root, dataset_list_filename), 'r') as dataset_list_file:
            dataset_files = dataset_list_file.readlines()

        return [
            (
                os.path.join(self.root, dataset_file.strip()),
                class_to_idx[os.path.basename(os.path.dirname(dataset_file))]
            )
            for dataset_file in dataset_files
        ]
