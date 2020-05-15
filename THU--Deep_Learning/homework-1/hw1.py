import mnist_data_loader
import matplotlib.pyplot as plt
import numpy as np
import sys
from softmax import Softmax


def example():
    example_id = 0
    image = train_set.images[example_id] # shape = 784 (28*28)
    label = train_set.labels[example_id] # shape = 1
    print(label)
    plt.imshow(np.reshape(image,[28,28]),cmap='gray')
    plt.show()

def plot_results(loss_hist, acc_hist, W):
    plt.plot(acc_hist)
    plt.title("Accuracy/Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("acc_hist2.png",bbox_inches='tight')
    plt.show()

    plt.plot(loss_hist)
    plt.title("Loss/Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_hist.png",bbox_inches='tight')
    plt.show()

    img_map_three = W[:,0]
    img_map_six   = W[:,1]

    plt.imshow(np.reshape(img_map_three,[28,28]),cmap='gray')
    plt.savefig("number_a.png", bbox_inches='tight')
    plt.show()

    plt.imshow(np.reshape(img_map_six,[28,28]),cmap='gray')
    plt.savefig("number_b.png", bbox_inches='tight')
    plt.show()


def normalize(y):
    y_norm = ( y - np.min(y) ) / (np.max(y) - np.min(y))
    return y_norm.astype(int)



if __name__ == "__main__":

    mnist_dataset = mnist_data_loader.read_data_sets("./MNIST_data/", one_hot=False)
    # training dataset
    train_set = mnist_dataset.train
    # train_set.labels = normalize(train_set.labels)
    # test dataset
    test_set = mnist_dataset.test
    # test_set.labels = normalize(test_set.labels)
    print("Training dataset size: ", train_set.num_examples)
    print("Test dataset size: ", test_set.num_examples)


    batch_size = 200
    max_epoch = 100
    reg = 1e-5

    loss_history = []
    acc_history = []
    classifier = Softmax(train_set.images.shape[1], len(np.unique(train_set.labels)) )

    for epoch in range(0, max_epoch):

        iter_per_batch = train_set.num_examples // batch_size

        
        for batch_id in range(0, iter_per_batch):
            # get the data of next minibatch (have been shuffled)
            batch = train_set.next_batch(batch_size)
            X, label = batch
            label = normalize(label)

            # Compute loss and gradient
            loss, grad = classifier.vectorized_loss(X, label, reg)
            loss_history.append(loss)

            # Generate Predictions
            y_train_pred = classifier.predict(X)
            y_train_acc = np.mean(y_train_pred == label)
            acc_history.append(y_train_acc)

            # update weights
            classifier.update_weights(grad)

            # print("ITER: {}, LOSS: {}, ACC: {}".format(batch_id,loss_history[-1],acc_history[-1]))

        print("ITER: {}, LOSS: {}, ACC: {}".format(epoch,loss_history[-1],acc_history[-1]))

    # Test Case
    print(">>> Computing the accuracy of the model on the test set.")

    y_test_label = normalize(test_set.labels)
    y_test_pred = classifier.predict(test_set.images)
    print("TEST SET ACC: {}".format( np.mean(y_test_pred == y_test_label) ))

    # Plot results
    plot_results(loss_history,acc_history,classifier.get_weights())