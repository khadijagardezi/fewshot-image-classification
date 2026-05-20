import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FewShotImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.filename_col = "filename"
        self.label_col = "label"

        self.classes = sorted(self.data[self.label_col].unique())

        self.class_to_images = {
            label: self.data[self.data[self.label_col] == label][self.filename_col].tolist()
            for label in self.classes
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = os.path.join(self.image_dir, row[self.filename_col])
        label = row[self.label_col]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label