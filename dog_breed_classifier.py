"""
CS5330 Final Project
Yueyang Wu, Yuyang Tian, Liqi Qi
"""

from utils import *

# allow using unverified SSL due to some configuration issue
ssl._create_default_https_context = ssl._create_unverified_context


def main(argv):
    """
    Build the training and testing data loaders
    Build and train the dog breeds classifier model
    Plot the results and save the model
    :param argv: code of the model to be trained ('m' : mobilenet, 'v' : vgg16, 'r' : resnet)
    """
    if len(argv) != 2:
        print('Wrong Input')
        return

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

    # build the model, initialize filename to save the final .pth file
    if argv[1] == 'm':
        model = MobilenetSubModel()
        filename = 'results/model_mobilenet.pth'
    elif argv[1] == 'v':
        model = VGG16Model()
        filename = 'results/model_vgg16.pth'
    elif argv[1] == 'r':
        model = ResNetSubModel()
        filename = 'results/model_resnet50.pth'

    # initialize the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train the model
    print('*****Model Info*****')
    print(f'Epoch Size: {N_EPOCHS}')
    print(f'Train Batch Size: {BATCH_SIZE_TRAIN}')
    print(f'Learning Rate: {LEARNING_RATE}')
    print('********************\n')

    accuracy_arr = []
    loss_arr = []
    start = datetime.now()
    train(train_loader=train_loader, model=model, loss_fn=loss_fn,
          optimizer=optimizer, accuracy_arr=accuracy_arr, loss_arr=loss_arr)
    end = datetime.now()

    print('Done!')
    print(f'Total Training Time in seconds: {(end - start).total_seconds()}')

    # save the final model
    torch.save(model.state_dict(), filename)

    # plot the accuracy and loss information
    plot_result(accuracy_arr, loss_arr)


if __name__ == "__main__":
    main(sys.argv)
