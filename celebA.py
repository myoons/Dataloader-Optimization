import os
import h5py
import math
import numpy as np
from tqdm import tqdm
from time import time
from PIL import Image


# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data.dataloader import Dataset, DataLoader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=18, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=18, out_channels=24, kernel_size=5)

        self.fc1 = nn.Linear(1512, 1)

        self.maxPool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        x = x.float()

        x = self.maxPool(F.relu(self.conv1(x)))
        x = self.maxPool(F.relu(self.conv2(x)))
        x = self.maxPool(F.relu(self.conv3(x)))
        x = self.maxPool(F.relu(self.conv4(x)))

        x = x.view(-1, 1512)
        x = F.relu(self.fc1(x))

        return F.sigmoid(x)


class CelebDataset(Dataset):
    def __init__(self, hdf5_path, img_key, labels, data_size):
        self.dataset = None
        self.labels = labels
        self.hdf5_path = hdf5_path
        self.img_key = img_key
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.hdf5_path, 'r', rdcc_nslots=11213, rdcc_nbytes=1024**2, rdcc_w0=1)[self.img_key]
            print('Setting Dataset')

        return self.dataset[idx], self.labels[idx]


def extract_gender_ndarray(annotation_list):
    gender_list = annotation_list[:, 20]
    for idx, gender_item in enumerate(gender_list):
        if gender_item == -1:
            gender_list[idx] = 0
    return np.array(gender_list)


def train_batch(batch_inputs, batch_targets, batch_size):

    input_a, input_b = torch.split(batch_inputs, math.ceil(batch_size / 2), dim=0)
    target_a, target_b = torch.split(batch_targets, math.ceil(batch_size / 2), dim=0)

    output_a = model(input_a)
    loss = criterion(output_a, target_a)
    loss.backward()
    optimizer.step()

    output_b = model(input_b)
    loss = criterion(output_b, target_b)
    loss.backward()
    optimizer.step()

    shuffle = torch.randperm(batch_size)
    batch_inputs = batch_inputs[shuffle]
    batch_targets = batch_targets[shuffle]

    return batch_inputs, batch_targets


if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    build_data = time()
    DATA_DIR = '/home/myoons/celeba_images_genders.h5'

    # Whether the file exists
    if os.path.isfile(DATA_DIR):  # File exists
        celebA = h5py.File(DATA_DIR, mode='r')
        images_h5 = celebA['images']  # (202559, 3, 218, 178)
        labels_h5 = celebA['labels']  # (202559,)
    else:  # Making H5 File
        celebA_raw = h5py.File('/data/CelebA/celeba_images_anno.h5', mode='r')
        celebA = h5py.File(DATA_DIR, 'w', rdcc_nslots=11213, rdcc_nbytes=1024**3, rdcc_w0=1)
        labels_h5 = extract_gender_ndarray(celebA_raw['anno'])  # (202559,)

        IMAGE_DATA_DIR = '/data/CelebA/img_align_celeba_png'
        IMAGE_ADDRESSES = [os.path.join(IMAGE_DATA_DIR, x) for x in os.listdir(IMAGE_DATA_DIR)]

        step = 20000
        start_idx = 0
        end_idx = start_idx + step
        size = len(IMAGE_ADDRESSES)
        batch_images = []

        while True:
            TARGET_ADDRESSES = IMAGE_ADDRESSES[start_idx:end_idx]

            for item in tqdm(TARGET_ADDRESSES, total=end_idx - start_idx):
                img = Image.open(item)
                img = np.array(img).transpose(2, 0, 1)  # (3, 218, 178) , uint8
                batch_images.append(img)

            batch_images = np.array(batch_images)

            if start_idx == 0:
                celebA.create_dataset('images',
                                      data=batch_images,
                                      dtype=np.uint8,
                                      chunks=(100, 3, 217, 178),  # 11 MB : Chunk Size
                                      maxshape=(None, 3, 218, 178))
            else:
                celebA['images'].resize((celebA['images'].shape[0] + batch_images.shape[0]), axis=0)
                celebA['images'][-batch_images.shape[0]:] = batch_images

            if end_idx == size:
                break

            start_idx += step
            end_idx = min(end_idx + step, size)
            batch_images = []

        celebA.create_dataset('labels',
                              data=labels_h5[:size],
                              dtype=np.uint8,
                              chunks=(20000,))

        images_h5 = celebA['images']  # (202559, 3, 218, 178)
        labels_h5 = celebA['labels']  # (202559,)
        celebA_raw.close()

    assert len(images_h5) == len(labels_h5), f"Must be SAME. Images Size : {len(images_h5)} / Labels : {len(labels_h5)}"

    print(f"""\n
            Building Dataset Completed / Time : {time() - build_data}
            Length : {len(images_h5)}
            Images : {images_h5}
            Genders : {labels_h5}
            """)

    labels_ndarray = np.array(labels_h5)
    celeb_dataset = CelebDataset(DATA_DIR, 'images', labels_ndarray, len(labels_h5))
    celeb_loader = DataLoader(dataset=celeb_dataset,
                              num_workers=1,
                              batch_size=512,
                              persistent_workers=True)
    celebA.close()
    model = CNN()
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    optimizer = torch.optim.Adam(params=parameters)
    criterion = nn.BCELoss()  # predictions, labels 순서
    model.to(device)
    criterion.to(device)

    summary(model, (3, 218, 178))
    model.train()

    for epoch in range(10):

        epoch_start = time()
        for batch_idx, (inputs, targets) in tqdm(enumerate(celeb_loader), total=len(celeb_loader)):

            inputs, targets = inputs.to(device), targets.float().to(device)
            targets = targets.view(-1, 1)  # (250, 1)
            batch_size = inputs.size(0)

            for i in range(2):
                inputs, targets = train_batch(inputs, targets, batch_size)

        print(f"Epoch : {epoch} Finished in {time() - epoch_start:.3f} Seconds")
