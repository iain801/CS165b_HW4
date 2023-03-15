print("Starting prediction.py")

from PIL import Image
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32

g = torch.Generator()
g.manual_seed(0)

print("Imports completed")

test_transforms = transforms.Compose([
    transforms.Resize(56),
    transforms.ToTensor(),
    transforms.Normalize((0.2103, 0.2103, 0.2103), (0.2961, 0.2961, 0.2961))  # normalize RGB channels with mean and std
])

#For Unlabeled Data
class UnlabeledDatasetFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        # Loop over each image file in the root directory and create a tuple (image_path, -1) for it
        for filenum in range(0,10000):
            filename = "{:d}.png".format(filenum)
            self.samples.append((os.path.join(root, filename), -1))
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, -1
    def __len__(self):
        return len(self.samples)
    

# Create a dataset and data loader for the test data
test_dir = './hw4_test'
test_dataset = UnlabeledDatasetFolder(root=test_dir, transform=test_transforms)

test_dataloader = DataLoader(test_dataset, 
    batch_size=64, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,)

# Load saved Model
savepath = './effnet_state.pth'

model = models.efficientnet_v2_l()
model.load_state_dict(torch.load(savepath))

# Instantiate the model and move it to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = model.to(device)

# Make predictions on the test data and write them to a file
model.eval()
with open('prediction.txt', 'w') as f:
    with torch.no_grad():
        for images, _ in test_dataloader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            for prediction in predicted:
                f.write(f'{prediction}\n')