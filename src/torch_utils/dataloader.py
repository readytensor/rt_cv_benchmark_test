import os
import random
import logging
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from config import paths
from utils import contains_subdirectories


class CustomDataLoader:
    OUTPUT_FOLDER = "output/image_output"
    MEAN = [0.485, 0.456, 0.406]
    STD_DEV = [0.229, 0.224, 0.225]

    def __init__(
        self,
        base_folder=paths.INPUTS_DIR,
        batch_size=256,
        num_workers=6,
        image_size=(224, 224),
        validation_size=0.0,
    ):
        self.base_folder = base_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = tuple(image_size)
        self.validation_size = validation_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.MEAN, std=self.STD_DEV),
            ]
        )
        (self.train_loader, self.test_loader, self.validation_loader) = (
            self.create_data_loaders()
        )

    def create_data_loaders(self):

        train_folder = os.path.join(self.base_folder, "training")
        test_folder = os.path.join(self.base_folder, "testing")
        validation_folder = os.path.join(self.base_folder, "validation")

        validation_exists = os.path.exists(
            validation_folder
        ) and contains_subdirectories(validation_folder)

        train_dataset = ImageFolder(root=train_folder, transform=self.transform)
        test_dataset = ImageFolder(root=test_folder, transform=self.transform)
        self.num_classes = len(train_dataset.classes)
        self.class_names = train_dataset.classes

        if self.validation_size > 0 and not validation_exists:
            train_dataset, validation_dataset = stratified_split_to_dataloaders(
                train_dataset, val_size=self.validation_size
            )
            print(len(train_dataset))
            validation_exists = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        validation_loader = None

        if validation_exists:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        return train_loader, test_loader, validation_loader

    @staticmethod
    def select_random_images(dataset, num_images=6):
        return random.sample(dataset.samples, num_images)

    @staticmethod
    def save_images(dataset_name, images):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
        for i, (image_path, _) in enumerate(images):
            img = Image.open(image_path)
            axes[i // 3, i % 3].imshow(img)
            axes[i // 3, i % 3].set_title(
                os.path.basename(image_path),
                fontweight="bold",
                fontname="Times New Roman",
            )
            axes[i // 3, i % 3].axis("off")
        plt.tight_layout()

        # Create the output directory if it doesn't exist
        os.makedirs(CustomDataLoader.OUTPUT_FOLDER, exist_ok=True)

        output_path = os.path.join(
            CustomDataLoader.OUTPUT_FOLDER,
            f'{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
        )
        plt.savefig(output_path)
        plt.close()

    def process_datasets(self):
        for dataset_name, loader in zip(
            ["train", "test"],
            [self.train_loader, self.test_loader],
        ):
            # Wrap the loader with tqdm for progress tracking
            for images, labels in tqdm(loader, desc=f"Processing {dataset_name} Data"):
                # Placeholder for processing code
                pass

            selected_images = self.select_random_images(loader.dataset)
            # Call save_images method to actually save the images
            self.save_images(dataset_name, selected_images)

            # Log progress
            for image_path, _ in selected_images:
                logging.info(
                    f"Saved {os.path.basename(image_path)} for {dataset_name} dataset"
                )


def stratified_split_to_dataloaders(dataset, val_size=0.1):
    # Ensure dataset.targets is available; if not, you might need to access dataset.targets in a different way depending on the dataset
    targets = np.array([dataset.targets[i] for i in range(len(dataset))])
    classes, class_counts = np.unique(targets, return_counts=True)
    class_indices = [np.where(targets == i)[0] for i in classes]

    train_indices, val_indices = [], []

    for indices in class_indices:
        np.random.shuffle(indices)
        split = int(len(indices) * val_size)
        val_indices.extend(indices[:split])
        train_indices.extend(indices[split:])

    # Creating the subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


# Script runs only when executed directly
if __name__ == "__main__":
    # Example usage
    data_loader = CustomDataLoader()
    data_loader.process_datasets()
