import os
import ssl

import pandas as pd
import torch
from PIL import Image
import torchvision
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# allow using unverified SSL due to some configuration issue
ssl._create_default_https_context = ssl._create_unverified_context


DATA_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/images'  # all images
LABEL_CSV_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/labels.csv'  # all images and labels
TRAIN_LABEL_CSV_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/train_data.csv'  # training images and labels
TEST_LABEL_CSV_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/test_data.csv'  # testing images and labels
N_EPOCHS = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
LEARNING_RATE = 0.003
LOG_INTERVAL = 10


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
        return len(self.breeds_frame)

    def __getitem__(self, idx):
        """
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


class ResNetSubModel(nn.Module):
    """
        PyTorch MobileNet Documentation: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
        Keep the features of MobileNet, modify the classifier
        """
    # initialize the model
    def __init__(self):
        super(ResNetSubModel, self).__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.model.fc = nn.Linear(2048, 120)
        # print('---------------model-------------')
        # print(self.model)

    def forward(self, x):
        return self.model(x)


def build_breed_code_dicts(csv_file):
    df = pd.read_csv(csv_file)
    label_arr = list(set(df.iloc[:, 1]))
    label_arr.sort()

    codes = range(len(label_arr))

    breed_to_code = dict(zip(label_arr, codes))
    code_to_breed = dict(zip(codes, label_arr))

    return breed_to_code, code_to_breed


def build_dataframe(csv_file, breed_to_code_dict):
    df = pd.read_csv(csv_file)
    df['code'] = [breed_to_code_dict[x] for x in df.breed]
    return df


def train(train_loader, test_loader, model, loss_fn, optimizer, n_epochs=N_EPOCHS):
    for epoch in range(n_epochs):
        size = len(train_loader.dataset)
        for batch, (X, y) in enumerate(train_loader):
            # compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # for each epoch, save a model version
        filename = 'results/model_resnet50_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), filename)

        test(test_loader=test_loader, model=model, loss_fn=loss_fn)


def test(test_loader, model, loss_fn=None):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    # make the code repeatable
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = True

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

    # plot the first image in the test dataset, for testing purpose
    # plt.imshow(train_dataset[0][0].permute(1, 2, 0))
    # plt.show()

    # load the MobileNet Model from PyTorch
    # mobilenet_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    # build ResNet model
    resnet_model = ResNetSubModel()

    # # print('-------------------mobilenet----------------------------')
    # # print(mobilenet_model)
    #
    # initialize the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=LEARNING_RATE)

    # train the model
    train(train_loader=train_loader, test_loader=test_loader, model=resnet_model, loss_fn=loss_fn, optimizer=optimizer)

    # # train the model
    # # for epoch in range(N_EPOCHS):
    # #     print(f'Epoch {epoch + 1}\n-------------------------------')
    # #     train(train_loader=train_loader, model=mobilenet_model, loss_fn=loss_fn, optimizer=optimizer)
    # #     if epoch % 10 == 0:
    # #         filename = 'results/model' + str(epoch) + '.pth'
    # #         torch.save(mobilenet_model.state_dict(), filename)
    # #     test(test_loader=test_loader, model=mobilenet_model, loss_fn=loss_fn)
    print('Done!')

    # save the final model
    torch.save(resnet_model.state_dict(), 'results/model_resnet50.pth')


if __name__ == "__main__":
    main()
