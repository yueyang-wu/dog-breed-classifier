import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import dog_breed_classifier


DATA_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/images'  # all images
LABEL_CSV_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/labels.csv'  # all images and labels
TRAIN_LABEL_CSV_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/train_data.csv'  # training images and labels
TEST_LABEL_CSV_PATH = '/Users/yueyangwu/Desktop/CS5330/final_proj/data/test_data.csv'  # testing images and labels


def main():
    mobilenet_model = dog_breed_classifier.MobilenetSubModel()
    mobilenet_model.load_state_dict(torch.load('results/model.pth'))
    mobilenet_model.eval()
    # print(mobilenet_model)

    # build breed and code convert dicts
    breed_to_code_dict, code_to_breed_dict = dog_breed_classifier.build_breed_code_dicts(LABEL_CSV_PATH)
    # print(breed_to_code_dict)

    # build dataframes
    train_df = dog_breed_classifier.build_dataframe(TRAIN_LABEL_CSV_PATH, breed_to_code_dict=breed_to_code_dict)
    test_df = dog_breed_classifier.build_dataframe(TEST_LABEL_CSV_PATH, breed_to_code_dict=breed_to_code_dict)

    # load the training and testing data
    # reshape the images to feed them to the model
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = dog_breed_classifier.DogBreedDataset(DATA_PATH, train_df, data_transform)
    test_dataset = dog_breed_classifier.DogBreedDataset(DATA_PATH, test_df, data_transform)

    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)

    dog_breed_classifier.test(test_loader=train_loader, model=mobilenet_model)

    # for data, target in train_loader:
    #     with torch.no_grad():
    #         output = mobilenet_model(data)
    #         print(output)
    #         code = output.argmax().item()
    #         print(code_to_breed_dict[code])


if __name__ == "__main__":
    main()
