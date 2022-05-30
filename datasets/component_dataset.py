import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ComponentDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.images = glob.glob(path + '/*.jpg')
        self.category_name2id = {"up": 0, "down": 1, "left": 2, "right": 3}
        
        category_name = self.path.split('/')[-1] # folder name
        self.category_id = self.category_name2id[category_name]

        self.data = []
        for image_path in self.images:
            self.data.append([image_path, self.category_id])
        self.image_dim = (32, 32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, category_id = self.data[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.image_dim)

        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2, 0, 1)
        
        return image_tensor, category_id


if __name__ == "__main__":
    train_data_dir = "data/sink/train/up"
    test_data_dir = "data/sink/valid/up"
    train_dataset = ComponentDataset(train_data_dir)
    test_dataset = ComponentDataset(test_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    for i, (image, category_id) in enumerate(train_loader):
        print(image.shape)
        print(category_id)
        if i == 0:
            break

    print(len(train_dataset))
