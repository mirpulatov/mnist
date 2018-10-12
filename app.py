import NeuralNetwork as Network
import pickle

# Loading data from pkl file
with open("mnist.pkl", "br") as fh:
    data = pickle.load(fh)
train_images = data[0]
test_images = data[1]
train_labels = data[2]
test_labels = data[3]
train_label = data[4]

NNet = Network.NeuralNetwork(input_layer=784, output_layer=10, hidden_layer=300, learning_rate=0.05)

epoch = 10

for i in range(1, epoch + 1):
    print("Epoch {0}:".format(i))
    for j in range(len(train_images)):
        NNet.train(train_images[j], train_label[j])

    # Evaluating
    corrects, wrongs = NNet.evaluate(train_images, train_labels)
    print("Accuracy on training set: ", corrects / (corrects + wrongs))
    corrects, wrongs = NNet.evaluate(test_images, test_labels)
    print("Accuracy on test set: ", corrects / (corrects + wrongs))
