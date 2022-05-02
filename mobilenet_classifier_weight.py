import os
import ssl

import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datetime import datetime
from dog_breed_classifier_mobilenet import *
import cv2 as cv

# allow using unverified SSL due to some configuration issue
ssl._create_default_https_context = ssl._create_unverified_context

DATA_PATH = '/Users/tian/Documents/Workspace/cs5330-final-project/data/images'  # all images
LABEL_CSV_PATH = '/Users/tian/Documents/Workspace/cs5330-final-project/data/labels.csv'  # all images and labels
TRAIN_LABEL_CSV_PATH = '/Users/tian/Documents/Workspace/cs5330-final-project/data/train_data.csv'  # training images and labels
TEST_LABEL_CSV_PATH = '/Users/tian/Documents/Workspace/cs5330-final-project/data/test_data.csv'  # testing images and labels
N_EPOCHS = 15
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
LEARNING_RATE = 0.001

def load_trained_model(model, pthFilePath):
    # load the previoud model and optimizer data from file
    model_state_dict = torch.load(pthFilePath)
    model.load_state_dict(model_state_dict)


def main():
    # make the code repeatable
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = False

    # build breed and code convert dicts
    breed_to_code_dict, code_to_breed_dict = build_breed_code_dicts(LABEL_CSV_PATH)

    # build dataframes
    train_df = build_dataframe(TRAIN_LABEL_CSV_PATH, breed_to_code_dict=breed_to_code_dict)
    test_df = build_dataframe(TEST_LABEL_CSV_PATH, breed_to_code_dict=breed_to_code_dict)

    # load the training and testing data
    # reshape the images to feed them to the model
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = DogBreedDataset(DATA_PATH, train_df, data_transform)
    test_dataset = DogBreedDataset(DATA_PATH, test_df, data_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)

    # build mobilenet model
    mobilenet_model = MobilenetSubModel()

    # load the previoud model from file
    load_trained_model(mobilenet_model, pthFilePath='/Users/tian/Documents/Workspace/cs5330-final-project/results/model_mobilenet_16_0.001.pth')

    print(mobilenet_model.model.classifier[0])


    #Analyze the classifier weight
    weight1 = mobilenet_model.model.classifier[1].weight
    print(weight1)
    print(weight1.shape)
    figure = plt.figure()
    plt.imshow(weight1.detach().numpy(), interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Show the effect of the filters
    training_sample = enumerate(train_loader)
    batch_idx, (training_sample_data, training_sample_targets) = next(training_sample)
    figure2 = plt.figure()
    plt.imshow(training_sample_data[0, 0].detach().numpy(), cmap='gray', interpolation='none')
    filtered_img = cv.filter2D(
        src=training_sample_data[0, 0].detach().numpy(),
        ddepth=-1,
        kernel=weight1.detach().numpy())
    plt.imshow(filtered_img, cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()
