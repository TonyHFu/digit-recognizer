from curses import flash
from flask import Flask, render_template, request, jsonify
import pandas as pd
import time
import base64
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
import re
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from joblib import load
import torch
import os

plt.switch_backend('agg')
# import cStringIO

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class MnistModel(nn.Module):
  def __init__(self, in_size, hidden_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, out_size)

  def forward(self, xb):
    xb = xb.view(xb.size(0), -1)
    out = self.linear1(xb)
    out = F.relu(out)
    out = self.linear2(out)
    return out
  
  def training_step(self, batch):
    images, labels = batch
    out = self(images)
    loss = F.cross_entropy(out, labels)
    return loss

  def validation_step(self, batch):
    images, labels = batch
    out = self(images)
    loss = F.cross_entropy(out, labels)
    acc = accuracy(out,labels)
    return {"val_loss": loss, "val_acc": acc}
  
    
  def validation_epoch_end(self, outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result["val_loss"], result["val_acc"]))




def b64_str_to_np(base64_str):
    
    base64_str = str(base64_str)
    if "base64" in base64_str:
        _, base64_str = base64_str.split(',')

    buf = BytesIO()
    buf.write(base64.b64decode(base64_str))
    buf.seek(0)
    pimg = Image.open(buf)
    img = np.array(pimg)

    # Keep only 4th value in 3rd dimension (first 3 are all zeros)
    return img[:, :, 3]

input_size = 784
num_classes = 10

model = MnistModel(input_size, hidden_size=32, out_size=num_classes)
model.load_state_dict(torch.load("model.pt"))
model.eval()

app = Flask(__name__)

@app.route('/digit', methods=["GET", "POST"])
def get_data():
  if request.method == "POST":
    # image_b64 = request.values['imageBase64']
    # image_data = re.sub('^data:image/.+;base64,', '', image_b64).decode('base64')
    # image_PIL = Image.open(cStringIO.StringIO(image_b64))
    # image_np = np.array(image_PIL)
    # print ('Image received: '.format(image_np.shape))
    
    
    data = request.get_json()["digit"]
    # print(type(data))
    # return "hello"
    # return data64
    img = np.fromstring(base64.b64decode(data[22:]), dtype=np.uint8)

    # img = base64.b64decode(data[22:])
    # image = Image.open(BytesIO(img))
    # image_gray = ImageOps.grayscale(image)
    # image_np = np.array(image_gray)
    # image_torch = torch.tensor(np.array(image))
    # resized_image = cv2.resize(image_np, (28, 28))
    
    # img = np.frombuffer(img, dtype=np.uint8).reshape((300, 300, 4))

    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img_gray = img[:, :, 3]
    resized_img = cv2.resize(img_gray, (28, 28)).astype('float32') 
    # img_inverted = cv2.bitwise_not(resized_img)
    
    # return render_template('index.html')
    # print(resized_character.shape)
    # ts = resized_character.tostring()
    # print(character.sum())
    # print("image mode", image.mode)
    # print("gray image", image_gray)
    # print(resized_image.shape)
    # print("sum is ", image_np.sum())
    # return image_np.tostring()
    # print("shape",img_inverted.shape)
    plt.imshow(resized_img, cmap="gray", vmin=0, vmax=255)
    plt.savefig("test2.png")
    plt.close("all")
    transform = transforms.ToTensor()

    # Convert the image to PyTorch tensor
    tensor = transform(resized_img)
    # print the converted image tensor
    # print(tensor)
    def predict_image(img, model):
      xb = img.unsqueeze(0)
      yb = model(xb)
      _, preds  = torch.max(yb, dim=1)
      print("Prediction:", preds[0].item())
      return preds[0].item()
    return str(predict_image(tensor, model))
  else:
    return "hello"
  


@app.route("/")
def home():
  return render_template('index.html')


