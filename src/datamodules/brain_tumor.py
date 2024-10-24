import os
import shutil
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import gdown
import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class BrainTumorDataModule(L.LightningModule):
    """
    A PyTorch Lightning DataModule for loading and preparing brain  tumor images
    for training, validation, and testing. This module manages the datasets
    and handles the dataloader configurations like batch size and number of workers.

    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        num_workers: int = 4,
        batch_size: int = 8,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None



    def split_dataset(self):
        data_path = self.data_dir / "dataset"
        train_path = self.data_dir / "train"
        val_path = self.data_dir / "validation"

        if not os.path.exists(train_path) or not os.path.exists(train_path):
            # create train and validation directories
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)

            # iterate through each class directory
            for class_dir in data_path.iterdir():
                if class_dir.is_dir() and not class_dir.name in ["train", "validation"]:
                    class_train = train_path / class_dir.name
                    class_test = val_path / class_dir.name

                    # create train and validation directories
                    os.makedirs(class_train, exist_ok=True)
                    os.makedirs(class_test, exist_ok=True)

                    # images
                    images = [
                        f
                        for f in class_dir.iterdir()
                        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                    ]

                    # train-validation files
                    train, test = train_test_split(
                        images, test_size=0.2, random_state=42
                    )

                    for file in train:
                        shutil.move(str(file), str(class_train))

                    for file in test:
                        shutil.move(str(file), str(class_test))

    def setup(self, stage: str):

        self.split_dataset()

        if stage == "fit" or stage is None:
            self.train_dataset = ImageFolder(
                root=self.data_dir / "train", transform=self.train_transform
            )

            self.val_dataset = ImageFolder(
                root=self.data_dir / "validation", transform=self.val_transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageFolder(
                root=self.data_dir / "validation", transform=self.val_transform
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return dataloader

    @property
    def normalize_transform(self):
        transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return transform

    @property
    def train_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )
        return transform

    @property
    def val_transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

        return transform
