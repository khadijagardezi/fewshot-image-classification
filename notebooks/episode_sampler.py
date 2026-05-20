import random
import os
import torch
from PIL import Image


class EpisodeSampler:
    def __init__(self, dataset, n_way=5, n_shot=5, n_query=5):
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

    def sample_episode(self):
        selected_classes = random.sample(
            list(self.dataset.class_to_images.keys()),
            self.n_way
        )

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for new_label, class_name in enumerate(selected_classes):
            image_files = self.dataset.class_to_images[class_name]

            selected_images = random.sample(
                image_files,
                self.n_shot + self.n_query
            )

            support_files = selected_images[:self.n_shot]
            query_files = selected_images[self.n_shot:]

            for img_file in support_files:
                img_path = os.path.join(self.dataset.image_dir, img_file)
                image = Image.open(img_path).convert("RGB")

                if self.dataset.transform:
                    image = self.dataset.transform(image)

                support_images.append(image)
                support_labels.append(new_label)

            for img_file in query_files:
                img_path = os.path.join(self.dataset.image_dir, img_file)
                image = Image.open(img_path).convert("RGB")

                if self.dataset.transform:
                    image = self.dataset.transform(image)

                query_images.append(image)
                query_labels.append(new_label)

        return (
            torch.stack(support_images),
            torch.tensor(support_labels),
            torch.stack(query_images),
            torch.tensor(query_labels),
            selected_classes
        )