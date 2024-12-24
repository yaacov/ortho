import os
from torch.utils.data import Dataset
from skimage import io

from data.glyph_classes import GLYPH_CLASSES


# Dataset class for loading images
class GlyphImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = GLYPH_CLASSES
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = io.imread(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
