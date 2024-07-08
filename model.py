import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO

def model(num_classes: int):
    model = YOLO("yolov8n-cls.pt")
    return model

    
def train(model, trainloader, optimizer, epochs, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    model.model.train()
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            ouputs = model.model(images)
            loss = criterion(ouputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Calculate and print average loss for the epoch
        epoch_loss = running_loss / len(trainloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

    
def test(model, testloader, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    model.model.eval()
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model.model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy