"""
CS5330 Final Project
Yueyang Wu, Yuyang Tian, Liqi Qi
"""

# import statements (do not delete the unused ones)
import os
import sys
import ssl
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datetime import datetime

# define hyper-parameters
DATA_PATH = './data/images'  # all images files
LABEL_CSV_PATH = './data/labels.csv'  # all image paths and labels
TRAIN_LABEL_CSV_PATH = './data/mini_train_data.csv'  # training images and labels
TEST_LABEL_CSV_PATH = './data/mini_test_data.csv'  # testing images and labels
N_EPOCHS = 15
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
LEARNING_RATE = 0.001


class DogBreedDataset(Dataset):
    """Dog Breed Dataset."""

    def __init__(self, root_dir, dataframe, transform=None):
        """
        :param root_dir (string) : Directory with all the images
        :param dataframe (pd.dataframe) : dataframe(id, breed, code)
        :param transform (callable, optional) : Optional transform to be applied on an sample
        """
        self.root_dir = root_dir
        self.breeds_frame = dataframe
        self.transform = transform

    def __len__(self):
        """
        :return: the length of the dataset
        """
        return len(self.breeds_frame)

    def __getitem__(self, idx):
        """
        :param idx: the index of the item
        :return sample ([image, breed]) : the sample at dataset[idx]
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.breeds_frame.iloc[idx, 0]) + '.jpg'
        image = Image.open(img_name)
        breed_code = self.breeds_frame.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        return [image, breed_code]


class MobilenetSubModel(nn.Module):
    """
    A deep network modified from the pretrained mobilenet v2 model from PyTorch
    PyTorch MobileNet Documentation: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    Keep the features of MobileNet, modify the classifier output to be 120
    """
    # initialize the model
    def __init__(self):
        super(MobilenetSubModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, 120)
        # print('---------------model-------------')
        # print(self.model)
        # self.features = model.features
        # print('----------------features-----------------')
        # print(self.features)

    # compute a forward pass
    def forward(self, x):
        return self.model(x)


class ResNetSubModel(nn.Module):
    """
    A deep network modified from the pretrained resnet50 model from PyTorch
    PyTorch ResNet50 Documentation: https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
    Keep the features of ResNet50, modify the fully connected layer output to be 120
    """
    # initialize the model
    def __init__(self):
        super(ResNetSubModel, self).__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.model.fc = nn.Linear(2048, 120)
        # print('---------------model-------------')
        # print(self.model)

    # compute a forward pass
    def forward(self, x):
        return self.model(x)


class VGG16Model(nn.Module):
    """
    A deep network modified from the pretrained vgg16 model from PyTorch
    PyTorch Vgg16 Documentation: https://pytorch.org/vision/main/generated/torchvision.models.vgg16.html
    Keep the features of Vgg16, modify the classifier output to be 120
    """
    # initialize the model
    def __init__(self):
        super(VGG16Model, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        for para in self.model.features.parameters():
            para.requires_grad = False
        self.model.classifier[-1] = nn.Linear(4096, 120)
        # print('---------------model-------------')
        # print(self.model)
        # self.features = model.features
        # print('----------------features-----------------')
        # print(self.features)

    # compute a forward pass
    def forward(self, x):
        return self.model(x)


def build_breed_code_dicts(csv_file):
    """
    Build a dictionary that matches the dog breeds string to numeric data
    :param csv_file: csv file with the dog breeds data
    :return: a dictionary of the dog breeds and numeric data
    """
    df = pd.read_csv(csv_file)
    label_arr = list(set(df.iloc[:, 1]))
    label_arr.sort()

    codes = range(len(label_arr))

    breed_to_code = dict(zip(label_arr, codes))
    code_to_breed = dict(zip(codes, label_arr))

    return breed_to_code, code_to_breed


def build_dataframe(csv_file, breed_to_code_dict):
    """
    Read a csv file with dog breeds data, convert the breed string to numeric data, build a dataframe
    :param csv_file: csv file with the dog breeds data
    :param breed_to_code_dict: dictionary to convert the string to numeric data
    :return: a dog breeds dataframe
    """
    df = pd.read_csv(csv_file)
    df['code'] = [breed_to_code_dict[x] for x in df.breed]
    return df


def train(train_loader, model, loss_fn, optimizer, accuracy_arr, loss_arr, n_epochs=N_EPOCHS):
    """
    Train the model
    :param train_loader: a data loader of training data
    :param model: the model to be trained
    :param loss_fn: the loss function used
    :param optimizer: the optimizer used
    :param accuracy_arr: the accuracy rate of each training epoch
    :param loss_arr: the average loss of each training epoch
    :param n_epochs: the number of training epochs
    """
    print('Before Train:')
    test(test_loader=train_loader, model=model, loss_fn=loss_fn, accuracy_arr=accuracy_arr, loss_arr=loss_arr)
    print('')
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch + 1}')
        size = len(train_loader.dataset)
        for batch, (X, y) in enumerate(train_loader):
            # compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"[{current:>5d}/{size:>5d}]")

        # for each epoch, save a model version
        filename = 'results/model_' + str(epoch + 1) + '.pth'
        torch.save(model.state_dict(), filename)

        # display the training accuracy and loss
        print('Train:')
        test(test_loader=train_loader, model=model, loss_fn=loss_fn, accuracy_arr=accuracy_arr, loss_arr=loss_arr)
        print('')


def test(test_loader, model, loss_fn, accuracy_arr, loss_arr):
    """
    Test the model performance
    :param test_loader: a data loader of the test data
    :param model: the model to be tested
    :param loss_fn: the loss function used
    :param accuracy_arr: the accuracy rate
    :param loss_arr: the average loss
    """
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            if loss_fn:
                test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    accuracy_arr.append(100 * correct)
    loss_arr.append(test_loss)


def plot_result(accuracy_arr, loss_arr):
    """
    Plot the accuracy rates and average losses
    :param accuracy_arr: an array of accuracy rates
    :param loss_arr: an array of average losses
    """
    x_axis = list(range(1, N_EPOCHS + 2))
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, accuracy_arr)
    plt.title('Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x_axis, loss_arr)
    plt.title('Loss')
    plt.tight_layout()
    plt.show()


def load_trained_model(model, pthFilePath):
    """
    load the previous model and optimizer data from file
    :param model: model to be loaded
    :param pthFilePath: pth file of the model
    """
    model_state_dict = torch.load(pthFilePath)
    model.load_state_dict(model_state_dict)
