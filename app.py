import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import numpy as np
import io


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

model = AutoEncoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("autoencoder_model.pt", map_location=torch.device("cpu")))
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



def generate_handwritten(letter):
    # Generate typed image
    img = Image.new('L', (28, 28), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    w, h = draw.textbbox((0, 0), letter, font=font)[2:]
    draw.text(((28 - w) / 2, (28 - h) / 2), letter, font=font, fill=0)

    # Transform and pass through model
    tensor_img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.encode(tensor_img)
        reconstructed = model.decode(z).squeeze().cpu().numpy()

    # Convert output tensor to image
    out_img = Image.fromarray((reconstructed * 255).astype('uint8').squeeze())
    resized_img = out_img.resize((280, 280), Image.NEAREST)  # 10x scale
    return resized_img


gr.Interface(
    fn=generate_handwritten,
    inputs=gr.Textbox(label="Type a character (A-Z)", lines=1, max_lines=1),
    outputs=gr.Image(label="Handwritten Output"),
    title="Handwriting Generator"
).launch()
