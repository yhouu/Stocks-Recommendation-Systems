import os
import os.path as p
import torch
import numpy as np
import random
import cv2
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

base_path='C:/cd_classifier'
root_dir='C:/cd_classifier/data/'
# base_path = os.getcwd()
# root_dir = p.join( base_path, 'data/' )

train_imgs = []
train_labels = []
test_imgs = []
test_labels = []


training_dir = p.join(root_dir, 'training_set/')
test_dir = p.join(root_dir, 'test_set/')


img_rows = 256
img_cols = 256


# Get train data
print("Get train & test data...")
for animal in ('cats/', 'dogs/'):
    animal_dir = p.join(training_dir, animal)
    os.chdir(animal_dir)
    files = os.listdir(animal_dir)

    for file in files:
        if '.jpg' in file: 
            f = cv2.imread(file, 0)
            f = torch.FloatTensor(cv2.resize(f, (img_rows, img_cols)))
        
        train_imgs.append(f)
        if animal == "cats/":
            train_labels.append(1)
        else:
            train_labels.append(0)
# Get test data
for animal in ('cats/', 'dogs/'):
    animal_dir = p.join(test_dir, animal)    
    os.chdir(animal_dir)
    files = os.listdir(animal_dir)

    for file in files:
        if '.jpg' in file: 
            f = cv2.imread(file, 0)
            f = torch.FloatTensor(cv2.resize(f, (img_rows, img_cols)))

        test_imgs.append(f)
        if animal == "cats/":
            test_labels.append(1)
        else:
            test_labels.append(0)
print("Finish\n")

train_imgs = torch.stack(train_imgs)
train_imgs = train_imgs.unsqueeze(1)
test_imgs = torch.stack(test_imgs)
test_imgs = test_imgs.unsqueeze(1)
train_labels = torch.LongTensor(train_labels)
test_labels = torch.LongTensor(test_labels)

# Split train, valid, test data
train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(train_imgs, train_labels, test_size=0.2, shuffle=True) 

# Make Dataset
train_ds = TensorDataset(train_imgs, train_labels)
valid_ds = TensorDataset(valid_imgs, valid_labels)
test_ds = TensorDataset(test_imgs, test_labels)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 1, 256, 256)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)   # Flatten them for FC
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)
        return logits

# Set gpu device
GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)

# Hyperparameter
BATCH_SIZE = 32
learning_rate = 0.00005
epochs = 10

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = None
ckpt_path = p.join(base_path, "ckpt/")
saved_ckpt_model = "checkpoint_epoch_0.pt"
train_loss = []

# Make Dataloader
print("Get Dataloader")
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True

)
valid_loader = DataLoader(
    dataset=valid_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)
print("Finish\n")

def save_ckpt(ckpt_path, model, optimizer, epoch, train_loss, best_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "train_loss" : train_loss,
        "best_loss" : best_loss
    }, ckpt_path)

def load_ckpt(ckpt_path, model, optimizer):
    try:
        checkpoint = torch.load(ckpt_path)
        print("Load checkpoint state.")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        best_loss = checkpoint['best_loss']
    except:
        print("No checkpoint state exist.")
        epoch = 0
        train_loss = []
        best_loss = 100
    
    return model, optimizer, epoch, train_loss, best_loss

model, optimizer, ckpt_epoch, train_loss, best_loss = load_ckpt(
        ckpt_path=p.join(ckpt_path, saved_ckpt_model), 
        model=model,
        optimizer=optimizer
    )

# Train
print("Start Training...")
os.makedirs(ckpt_path, exist_ok=True)
best_epoch = 0
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for step, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        
        loss = criterion(logits.squeeze(1), labels.float())
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch + 1}],  loss: {epoch_loss/(step + 1): .3f}")

    train_loss.append(epoch_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        current_loss = 0.0
        for step, (inputs, labels) in enumerate(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
        
            loss = criterion(logits.squeeze(1), labels.float())
            current_loss += loss.item()
        valid_loss = current_loss / len(valid_loader)

    if valid_loss < best_loss:
            best_loss = valid_loss
            save_ckpt(
                ckpt_path=p.join(ckpt_path, f"checkpoint_epoch_{epoch + 1}.pt"), 
                model=model, optimizer=optimizer,
                epoch=epoch+1,
                train_loss=train_loss, best_loss=valid_loss
            )
            best_epoch = epoch + 1
            print(f"Success to save checkpoint. Best loss so far: {best_loss: .3f}")
    else:
        print("No improvement detected. Skipping save")
print(f"Best epoch : {best_epoch}")
print("Finish\n")

# Test
print("Start Test...")

test_correct = 0
test_total = 0

test_model = CNN()
test_model, optimizer, ckpt_epoch, train_loss, best_loss = load_ckpt(
        ckpt_path=p.join(ckpt_path, f"checkpoint_epoch_{best_epoch}.pt"), 
        model=model,
        optimizer=optimizer
    )
test_model = test_model.to(device)
test_model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = test_model(inputs)
        
        loss = criterion(logits.squeeze(1), labels.float())
        pred = torch.sigmoid(logits.squeeze(1))
        pred = torch.round(pred)
        test_total += labels.size(0)
        test_correct += (pred == labels).sum().item()
test_acc = test_correct / test_total

test_acc = test_acc * 100
print('TestAcc: %.2f' % (test_acc))

print("All finished")


