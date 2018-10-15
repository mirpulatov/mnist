import NeuralNetwork as Network
import pickle
import gzip

f = gzip.open('mnist.pkl.gz', 'rb')
data = pickle.load(f)
train_images = data[0]
test_images = data[1]
train_labels = data[2]
test_labels = data[3]
train_label = data[4]
f.close()

NNet = Network.NeuralNetwork(input_layer=784, output_layer=10, hidden_layer=300, learning_rate=0.03)

epoch = 15

for i in range(1, epoch + 1):
    print("Epoch {0}:".format(i))
    for j in range(len(train_images)):
        NNet.train(train_images[j], train_label[j])

    # Evaluating
    corrects, wrongs = NNet.evaluate(train_images, train_labels)
    print("Accuracy on training set: ", corrects / (corrects + wrongs))
    corrects, wrongs = NNet.evaluate(test_images, test_labels)
    print("Accuracy on test set: ", corrects / (corrects + wrongs))

