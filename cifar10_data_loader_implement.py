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
import math
import queue
import multiprocessing
import itertools


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


# TODO: 구조가 살짝 비효율적이라는 생각이 든다
class CifarDataset(Dataset):
    def __init__(self, images, labels):
        assert len(images) == len(labels), "Length of images and labels are different!"
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


class CifarDataLoader(object):

    def __init__(self, dataset, batch_size=64, num_workers=1, prefetch_batches=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0  # Index of one data (image, label)

        self.num_workers = num_workers  # Number of workers
        self.prefetch_batches = prefetch_batches  # Number of prefetch batches per worker
        self.prefetch_index = 0  # Index of the item that should be prefetched next
        self.output_queue = multiprocessing.Queue()  # Queue that is shared across all of the worker processes

        self.index_queues = []
        self.workers = []  # List of workers (multiprocess.Process)
        self.worker_cycle = itertools.cycle(range(num_workers))
        self.cache = {}

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()  # Queue that each worker owns
            worker = multiprocessing.Process(target=worker_fn, args=(self.dataset, index_queue, self.output_queue))
            worker.daemon = True  # Daemon process
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self.prefetch()  # Keep adding indices to each workers queue until number of prefetch_batches are added

    def __iter__(self):
        self.index = 0
        self.cache = {}
        self.prefetch_index = 0
        self.prefetch()
        # TODO: 여기 Shuffle 있어야 한다
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration  # stop iteration once index is out of bounds

        images = []
        labels = []
        for idx in range(self.index, min(self.index + self.batch_size, len(self.dataset))):

            image, label = self.get()
            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __del__(self):
        try:
            # Stop each worker by passing None to its index queue
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:  # close all queues
                q.cancel_join_thread()
                q.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():  # manually terminate worker if all else fails
                    w.terminate()

    def prefetch(self):
        # TODO: What does self.prefetch_index < self.index + 2 * self.num_workers * self.batch_size means?
        while self.prefetch_index < len(self.dataset) and self.prefetch_index < self.index + 2 * self.num_workers * self.batch_size:
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += 1

    def get(self):
        self.prefetch()

        if self.index in self.cache:
            item = self.cache[self.index]
            del self.cache[self.index]
        else:
            while True:
                try:
                    (index, data) = self.output_queue.get(timeout=0)
                except queue.Empty:
                    continue
                if index == self.index:
                    item = data
                    break
                else:
                    self.cache[index] = data

        self.index += 1
        return item


def worker_fn(dataset, index_queue, output_queue):
    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:  # Exception occurred, No item in Queue
            break
        output_queue.put((index, dataset[index]))


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

    train_set = CifarDataset(images=train_images, labels=train_labels)
    train_loader = CifarDataLoader(dataset=train_set, batch_size=128, num_workers=4)

    print(f"""\n
    Loading Train Dataset Completed / Time : {time() - train_time}
    Length : {len(train_images)}
    Shape : {train_images.shape}
    Type : {type(train_images)}
    """)

    model_time = time()
    model = CNN()
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optimizer = torch.optim.Adam(params=parameters)
    print(f'\nModel on GPU / Time : {time() - model_time}')

    criterion = nn.CrossEntropyLoss()
    model.to(device)

    summary(model, (3, 32, 32))

    print('\nStart Training\n')
    for epoch in range(100):
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = torch.from_numpy(inputs)  # Size : torch.Tensor([3, 32, 32])
            targets = torch.tensor(targets, dtype=torch.long)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute the Loss
            loss.backward()  # Compute the Gradients
            optimizer.step()

        print(f'{epoch} Finished\n')
