from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import io

app = FastAPI()

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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load image from the uploaded file
    image = Image.open(io.BytesIO(await file.read()))
    image_width, image_height = image.size

    # Initialize and load the trained model based on the uploaded image dimensions
    model = CarPricePredictor(image_height, image_width)
    model.load_state_dict(torch.load(r"C:\Users\alvar\Downloads\car_price_model.pth"))
    model.eval()

    # Prepare the image tensor
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).float()

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
    price = output.item()

    return JSONResponse(content={"Predicted Price": f"${price:,.2f}"})
