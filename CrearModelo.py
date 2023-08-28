import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

# Custom Dataset class to load images and prices
class CarDataset(Dataset):
    def __init__(self, h5file_path):
        self.h5file = h5py.File(h5file_path, 'r')
        self.data = []
        for page_key in self.h5file.keys():
            page_group = self.h5file[page_key]
            for car_key in page_group.keys():
                car_group = page_group[car_key]
                self.data.append((car_group['Image'][:], car_group.attrs['Price']))
        # Get the shape of the first image to determine the dimensions
        self.image_shape = self.data[0][0].shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_array, price = self.data[idx]
        return transforms.ToTensor()(Image.fromarray(image_array)), float(price)

# Load Data
dataset = CarDataset('car_data.h5')
image_height, image_width, _ = dataset.image_shape

# Neural Network Architecture
class CarPricePredictor(nn.Module):
    def __init__(self, input_height, input_width):
        super(CarPricePredictor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Dummy forward pass to calculate the flattened size
        x = torch.zeros(1, 3, input_height, input_width)
        x = self.conv_layers(x)
        self.flattened_size = x.view(-1).size(0)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Model, Loss, and Optimizer
model = CarPricePredictor(image_height, image_width)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for epoch in range(10):
    for images, prices in train_dataloader:
        images = images.float() # Cast images to Float
        prices = prices.float().view(-1, 1) # Cast prices to Float
        outputs = model(images)
        loss = criterion(outputs, prices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate RMSE on Test Data
total_squared_error = 0
total_count = 0
with torch.no_grad():
    for images, prices in test_dataloader:
        images = images.float() # Cast images to Float
        prices = prices.float() # Cast prices to Float
        outputs = model(images)
        squared_error = (outputs.view(-1) - prices) ** 2
        total_squared_error += squared_error.sum().item()
        total_count += len(prices)

rmse = np.sqrt(total_squared_error / total_count)
print(f'Root Mean Squared Error on Test Data: RD${rmse:,.2f}')

# Save the Model
torch.save(model.state_dict(), 'car_price_model.pth')
