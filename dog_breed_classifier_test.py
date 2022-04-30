from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import dog_breed_classifier_mobilenet
import dog_breed_classifier_resnet50
import dog_breed_classifier_vgg16


DATA_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/images'  # all images
LABEL_CSV_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/labels.csv'  # all images and labels
TRAIN_LABEL_CSV_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/train_data.csv'  # training images and labels
TEST_LABEL_CSV_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/test_data.csv'  # testing images and labels


def main():
    # load mobilenet model
    mobilenet_model = dog_breed_classifier_mobilenet.MobilenetSubModel()
    mobilenet_model.load_state_dict(torch.load('results/model_mobilenet.pth'))
    mobilenet_model.eval()
    # print(mobilenet_model)

    # load resnet model
    resnet_model = dog_breed_classifier_resnet50.ResNetSubModel()
    resnet_model.load_state_dict(torch.load('results/model_resnet50.pth'))
    resnet_model.eval()

    # load vgg16 model
    vgg16_model = dog_breed_classifier_vgg16.VGG16Model()
    vgg16_model.load_state_dict(torch.load('results/model_vgg16.pth'))

    # load mobilenet experiment model
    mobilenet_model_epoch10_lr0001 = dog_breed_classifier_mobilenet.MobilenetSubModel()
    mobilenet_model_epoch10_lr0001.load_state_dict(torch.load('results/model_mobilenet_8_0.001.pth'))

    # build breed and code convert dicts
    breed_to_code_dict, code_to_breed_dict = dog_breed_classifier_mobilenet.build_breed_code_dicts(LABEL_CSV_PATH)
    # print(breed_to_code_dict)

    # build dataframes
    train_df = dog_breed_classifier_mobilenet.build_dataframe(TRAIN_LABEL_CSV_PATH, breed_to_code_dict=breed_to_code_dict)
    test_df = dog_breed_classifier_mobilenet.build_dataframe(TEST_LABEL_CSV_PATH, breed_to_code_dict=breed_to_code_dict)

    # load the training and testing data
    # reshape the images to feed them to the model
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = dog_breed_classifier_mobilenet.DogBreedDataset(DATA_PATH, train_df, data_transform)
    test_dataset = dog_breed_classifier_mobilenet.DogBreedDataset(DATA_PATH, test_df, data_transform)

    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    accuracy_arr = []
    loss_arr = []
    start = datetime.now()

    # test on mobilenet
    # dog_breed_classifier_mobilenet.test(test_loader=test_loader, model=mobilenet_model, loss_fn=loss_fn,
    # accuracy_arr=accuracy_arr, loss_arr=loss_arr)

    # test on resnet
    # dog_breed_classifier_mobilenet.test(test_loader=test_loader, model=resnet_model, loss_fn=loss_fn,
    #                                     accuracy_arr=accuracy_arr, loss_arr=loss_arr)

    # test on vgg16
    # dog_breed_classifier_mobilenet.test(test_loader=test_loader, model=vgg16_model, loss_fn=loss_fn,
    #                                     accuracy_arr=accuracy_arr, loss_arr=loss_arr)

    # test on mobilenet
    dog_breed_classifier_mobilenet.test(test_loader=test_loader, model=mobilenet_model_epoch10_lr0001, loss_fn=loss_fn,
                                        accuracy_arr=accuracy_arr, loss_arr=loss_arr)

    end = datetime.now()
    print(f'Total Testing Time in seconds: {(end - start).total_seconds()}')


if __name__ == "__main__":
    main()
