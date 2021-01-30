import os
import torch
import numpy as np
from time import time
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data.dataloader import Dataset, DataLoader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=15, kernel_size=3)

        self.fc1 = nn.Linear(540, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

        self.maxPool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        x = x.float()

        x = self.maxPool(F.relu(self.conv1(x)))  # (6, 15, 15)
        x = self.maxPool(F.relu(self.conv2(x)))  # (15, 6, 6)
        x = x.view(-1, 540)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return F.softmax(x, dim=1)


class CifarDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, data_size: int):
        self.images = images
        self.labels = labels
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image: np.ndarray = self.images[idx]
        label: np.uint8 = self.labels[idx]
        return image, label


def open_image_attach_label_tuple(data_address, label_map):
    image = Image.open(data_address)
    image = np.array(image)
    image = image.transpose(2, 0, 1)
    label_name = data_address[:-4].split("_")[-1]
    label = label_map[label_name]
    return image, label


def images_to_image_label_ndarray(data_dir, label_map):
    data_addresses = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    images, labels = zip(*[open_image_attach_label_tuple(data_address, label_map) for data_address in data_addresses])
    image_ndarray = np.array(images, dtype=np.uint8)
    labels_ndarray = np.array(labels, dtype=np.uint8)
    return image_ndarray, labels_ndarray


if __name__ == '__main__':

    with open("cifar/labels.txt") as label_file:
        class_labels = label_file.read().split()
        label_mapping = dict(zip(class_labels, list(range(len(class_labels)))))

    train_time = time()

    if os.path.isfile('train_images.npy') and os.path.isfile('train_labels.npy'):
        train_images = np.load('train_images.npy', allow_pickle=True)
        train_labels = np.load('train_labels.npy', allow_pickle=True)
    else:
        train_images, train_labels = images_to_image_label_ndarray(data_dir="cifar/train/", label_map=label_mapping)
        np.save('train_images.npy', train_images)
        np.save('train_labels.npy', train_labels)

    assert len(train_images) == len(train_labels), "Length of images and labels should be same!"
    train_set = CifarDataset(images=train_images, labels=train_labels, data_size=len(train_images))
    train_loader = DataLoader(dataset=train_set,
                              num_workers=4,
                              batch_size=512,
                              persistent_workers=True,
                              )
    print(f"""\n
    Loading Train Dataset Completed / Time : {time() - train_time}
    Length : {len(train_images)}
    Shape : {train_images.shape}
    Type : {type(train_images)}
    """)

    model_time = time()
    model = CNN()
    model = nn.DataParallel(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    optimizer = torch.optim.Adam(params=parameters)
    criterion = nn.CrossEntropyLoss().to(device);
    model.to(device)

    summary(model, (3, 32, 32))

    print('\nStart Training\n')
    start = time()
    for epoch in range(180):

        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, targets = inputs.to(device, non_blocking=True), targets.long().to(device, non_blocking=True)

            outputs = model(inputs)  # Forward pass, GPU
            loss = criterion(outputs, targets)  # Compute the Loss, GPU
            loss.backward()  # Compute the Gradients, GPU
            optimizer.step()

    print(time()-start)
