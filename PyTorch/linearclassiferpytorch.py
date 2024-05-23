pip install skillsnetwork

from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim
import skillsnetwork

await skillsnetwork.prepare("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip", path = "resources/data", overwrite=True)

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="resources/data"
        positive="Positive"
        negative="Negative"

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
        positive_files.sort()
        negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
        negative_files.sort()
        number_of_samples=len(positive_files)+len(negative_files)
        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0

        if train:
            self.all_files=self.all_files[0:10000] #Change to 30000 to use the full test dataset
            self.Y=self.Y[0:10000] #Change to 30000 to use the full test dataset
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)

    # Get the length
    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):


        image=Image.open(self.all_files[idx])
        y=self.Y[idx]


        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# transforms.ToTensor()
#transforms.Normalize(mean, std)
#transforms.Compose([])

transform =transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean, std)])

dataset_train=Dataset(transform=transform,train=True)
dataset_val=Dataset(transform=transform,train=False)

# dataset_train[0][0].shape

size_of_image=3*227*227


"""
   learning rate:0.1 
   momentum term:0.1 
   batch size training:5
   Loss function:Cross Entropy Loss
   epochs:5</li>
   set: torch.manual_seed(0)
"""

torch.manual_seed(0)

# Custom Module

class Model(nn.Module):
  def __init__(self, size_of_image):
    super(Model, self).__init__()
    self.flatten = nn.Flatten()  # Flatten the input image
    self.linear = nn.Linear(size_of_image, 2)  # Linear layer for 2 classes
    self.softmax = nn.Softmax(dim=1)  # Softmax for probability distribution

    self.best_val_accuracy = 0.0  # Initialize best validation accuracy

  def forward(self, x):
    x = self.flatten(x)
    x = self.linear(x)
    x = self.softmax(x)  # Apply Softmax for classification probabilities
    return x

  def train_model(self, train_loader, val_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
      # Training loop
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = self(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

      # Validation loop
      val_correct = 0
      val_total = 0
      with torch.no_grad():
        for images, labels in val_loader:
          outputs = self(images)
          _, predicted = torch.max(outputs.data, 1)
          val_total += labels.size(0)
          val_correct += (predicted == labels).sum().item()

      val_accuracy = val_correct / val_total
      if val_accuracy > self.best_val_accuracy:
        self.best_val_accuracy = val_accuracy

      print(f'Epoch [{epoch+1}/{epochs}], Validation Accuracy: {val_accuracy:.4f}')

    print(f'Best Validation Accuracy Achieved: {self.best_val_accuracy:.4f}')
      
# Model Object
model = model(size_of_image)

# Optimizer
learning_rate = 0.01
momentum_term = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_term)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Data Loader Training and Validation
batch_size = 5
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size)

# Train Model
model.train_model(train_loader, val_loader, optimizer, criterion, epochs=5)

