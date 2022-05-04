"""
CS5330 Final Project
Yueyang Wu, Yuyang Tian, Liqi Qi
"""

from utils import *


def main(argv):
    """
    Load a model
    Build the training and testing data loaders
    Apply the model and display the results
    :param argv: code of the model to be tested ('m' : mobilenet, 'v' : vgg16, 'r' : resnet)
    """
    if len(argv) != 2:
        print('Wrong Input')
        return

    # load model
    if argv[1] == 'm':
        model = MobilenetSubModel()
        model.load_state_dict(torch.load('results/model_mobilenet.pth'))
    elif argv[1] == 'r':
        model = ResNetSubModel()
        model.load_state_dict(torch.load('results/model_resnet50.pth'))
    elif argv[1] == 'v':
        model = VGG16Model()
        model.load_state_dict(torch.load('results/model_vgg16.pth'))
    else:  # load experimental models (change the model accordingly)
        model = MobilenetSubModel()
        model.load_state_dict(torch.load('results/model_mobilenet_8_0.001.pth'))
    model.eval()

    # build breed and code convert dicts
    breed_to_code_dict, code_to_breed_dict = build_breed_code_dicts(LABEL_CSV_PATH)

    # build dataframes
    train_df = build_dataframe(TRAIN_LABEL_CSV_PATH, breed_to_code_dict=breed_to_code_dict)
    test_df = build_dataframe(TEST_LABEL_CSV_PATH, breed_to_code_dict=breed_to_code_dict)

    # load the training and testing data
    # reshape the images to feed them to the model
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = DogBreedDataset(DATA_PATH, train_df, data_transform)
    test_dataset = DogBreedDataset(DATA_PATH, test_df, data_transform)

    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    accuracy_arr = []
    loss_arr = []
    start = datetime.now()

    # test on the model
    test(test_loader=test_loader, model=model, loss_fn=loss_fn, accuracy_arr=accuracy_arr, loss_arr=loss_arr)

    end = datetime.now()
    print(f'Total Testing Time in seconds: {(end - start).total_seconds()}')


if __name__ == "__main__":
    main(sys.argv)
