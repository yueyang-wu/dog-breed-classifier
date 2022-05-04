"""
CS5330 Final Project
Yueyang Wu, Yuyang Tian, Liqi Qi
"""

from utils import *

# allow using unverified SSL due to some configuration issue
ssl._create_default_https_context = ssl._create_unverified_context

# define hyper-parameters
N_EPOCHS = 10
# BATCH_SIZE = [8, 16, 32]
batch_size = 8
LEARNING_RATE = [0.001, 0.1, 1]


def plot_result_with_label(accuracy_arr, loss_arr, label):
    """
    Plot the accuracy rates and average losses with a given label
    :param accuracy_arr: an array of accuracy rates
    :param loss_arr: an array of average losses
    :param label: the label for the plot
    """
    x_axis = list(range(1, N_EPOCHS + 2))
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, accuracy_arr, label=label)
    plt.title('Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x_axis, loss_arr, label=label)
    plt.title('Loss')
    plt.tight_layout()
    plt.legend(loc='best')
    return plt


def main():
    """
    Build the training and testing data loaders
    Build and train several different MobileNet models, modifying the training batch size and the learning rate
    Plot the results and save the models
    """
    # make the code repeatable
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = False

    # build mobilenet model
    mobilenet_model = MobilenetSubModel()

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

    for lr in LEARNING_RATE:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # initialize the loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(mobilenet_model.parameters(), lr=lr)

        # train the model
        print('*****Model Info*****')
        print(f'Epoch Size: {N_EPOCHS}')
        print(f'Train Batch Size: {batch_size}')
        print(f'Learning Rate: {lr}')
        print('********************\n')

        accuracy_arr = []
        loss_arr = []
        start = datetime.now()
        train(train_loader=train_loader, model=mobilenet_model, loss_fn=loss_fn,
              optimizer=optimizer, accuracy_arr=accuracy_arr, loss_arr=loss_arr, n_epochs=N_EPOCHS)
        end = datetime.now()

        print('Done!')
        print(f'Total Training Time in seconds: {(end - start).total_seconds()}')

        # save the final model
        filename = f'results/model_mobilenet_{batch_size}_{lr}.pth'
        torch.save(mobilenet_model.state_dict(), filename)

        # plot the accuracy and loss information
        label = f'{batch_size}_{lr}'
        plot = plot_result_with_label(accuracy_arr, loss_arr, label)
        plot_name = f'results/mobilenet_experiment.png'
        plot.savefig(plot_name)


if __name__ == "__main__":
    main()
