print("Starting training.py")

from PIL import Image
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

from copy import deepcopy
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

# Define the transforms for the training and validation datasets
train_transforms = transforms.Compose([
    transforms.Resize(56),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.2103, 0.2103, 0.2103), (0.2961, 0.2961, 0.2961)) # normalize RGB channels with mean and std
])

# Define the dataset
data_dir = './hw4_train'
dataset = ImageFolder(data_dir, transform=train_transforms)

print(f"Classes: {dataset.classes}")

# Define the validation split as 10% of the data
val_size = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Training set: {train_size}")
print(f"Validation set: {val_size}")

# Define the dataloaders
train_dataloader = DataLoader(train_dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,)
val_dataloader = DataLoader(val_dataset, 
    batch_size=64, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,)

print("Datasets loaded")

# Instantiate the model and move it to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = models.efficientnet_v2_l(weights='DEFAULT')

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
  
model = model.to(device)

print("Model Defined")

learnrate = 0.005

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learnrate)
scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)

# Train the model
best_val_acc = 0.0
best_model_state = None

num_epochs = 50
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for i, data in enumerate(train_dataloader, 0):
        # Get inputs and labels from dataloader
        inputs, labels = data
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)

    train_loss /= len(train_dataloader.dataset)
    train_acc = train_acc.double() / len(train_dataloader.dataset)
            
    val_loss = 0.0
    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        for data in val_dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True) # move inputs to device
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)
            
    val_loss /= len(val_dataloader.dataset)
    val_acc = val_acc.double() / len(val_dataloader.dataset)

    # update the learning rate scheduler
    scheduler.step(val_loss)

    # check if the model's validation accuracy has improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = deepcopy(model.state_dict())

    # print epoch statistics
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_model_state)

print('Finished Training')

# Evaluate the model on the test set
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in val_dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True) # move inputs to device
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = (correct / total) * 100
print(f"Accuracy on validation set: {val_accuracy:.3f}%")

savepath = './effnet_state.pth'
torch.save(model.module.state_dict(), savepath)
