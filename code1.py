import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_data(train_path, test_path):
    def one_hot(Y_train):
        temp = np.zeros((Y_train.shape[0], 10))
        for i in range(Y_train.shape[0]):
            temp[i, int(Y_train[i])] = 1
        return temp
    def shuffle():
        pass
    # temp_train = np.loadtxt(train_path, delimiter = ',' , skiprows=1)
    # with open("temp_train.p", 'wb') as h:
    #     pickle.dump(temp_train, h,protocol=pickle.HIGHEST_PROTOCOL)
    with open("temp_train.p", "rb") as h:
        temp_train = pickle.load(h)
    X_train = temp_train[:, 1:]
    Y_train = temp_train[:, 0]
    X_train = X_train / np.max(X_train) - np.min(X_train)
    Y_train = one_hot(Y_train)
    # temp_test = np.loadtxt('./mnist_test.csv', delimiter = ',' , skiprows=1)
    # with open("temp_test.p", 'wb') as h:
    #     pickle.dump(temp_test, h,protocol=pickle.HIGHEST_PROTOCOL)
    with open("temp_test.p", "rb") as h:
        temp_test = pickle.load(h)

    X_test = temp_test[:, 1:]
    Y_test = temp_test[:, 0]
    X_test = X_test / np.max(X_test) - np.min(X_test)
    Y_test = one_hot(Y_test)
    return X_train, Y_train, X_test, Y_test

def preprocess_data():
    #normalize
    pass

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return 1 * (x > 0)

#Softmax
def softmax(z):
    z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
    return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)

#derivative of softmax
def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

def init_neural_network( X_train, Y_train, X_test, Y_test):
    num_unit_layer_1 = 256
    num_unit_layer_2 = 128
    theta1 = np.random.random_sample((X_train.shape[1], num_unit_layer_1))
    theta2 = np.random.random_sample((num_unit_layer_1, num_unit_layer_2))
    theta3 = np.random.random_sample((num_unit_layer_2, Y_train.shape[1]))
    b1 = np.random.random_sample((num_unit_layer_1))
    b2 = np.random.random_sample((num_unit_layer_2))
    b3 = np.random.random_sample((Y_train.shape[1]))
    batch = 64
    lr = 5e-4
    epochs = 1000

    #forward
    for epoch in range( epochs):
        error_train = 0
        acc = 0
        num_samples = X_train.shape[0] - 59000
        # num_samples = 200
        for b in range(num_samples//batch-1):
            start = b*batch
            end = (b+1)*batch
            a1 = X_train[start:end]
            z2 = a1 @ theta1
            a2 = relu(z2)
            z3 = a2 @ theta2
            a3 = relu(z3)
            z4 =  a3 @ theta3
            a4 = softmax(z4)

            # Backprop
            err = (a4 - Y_train[start:end])
            delta4 = (err / batch)
            delta3 = (delta4@theta3.T) * drelu(z3) # dsoftmax
            delta2 = (delta3@theta2.T) * drelu(z2)
            theta3 -= (lr * (delta4.T @ a3).T)
            theta2 -= (lr * (delta3.T @ a2).T)
            theta1 -= (lr * (delta2.T @ a1).T)
            # db3 = np.sum(delta4,axis = 0)
            # db2 = np.sum((delta4@theta3.T) * drelu(z3), axis=0)
            # db1 = np.sum((db2@theta2.T) * drelu(z2), axis=0)
            # b3 = b3 - (lr * db3)
            # b2 = b2 - (lr * db2)
            # b1 = b1 - (lr * db1)

            error_train += np.mean(err**2)
            acc+= np.count_nonzero(np.argmax(a4,axis=1) == np.argmax(Y_train[start:end],axis=1)) / batch

        error_train = error_train/(num_samples//batch-1)
        error_train = "{:.3f}".format(error_train)

        acc = acc/(num_samples//batch-1)*100
        acc = "{:.1f}".format(acc)

        #test data
        if epoch % 10 ==0:
            acc_test = 0
            for b in range(X_test.shape[0] // batch - 1):
                start = b * batch
                end = (b + 1) * batch
                a1 = X_test[start:end]
                z2 = a1 @ theta1
                a2 = relu(z2)
                z3 = a2 @ theta2
                a3 = relu(z3)
                z4 = a3 @ theta3
                a4 = softmax(z4)
                acc_test += np.count_nonzero(np.argmax(a4, axis=1) == np.argmax(Y_test[start:end], axis=1)) / batch
            acc_test = acc_test / (X_test.shape[0] // batch - 1) * 100
            acc_test = "{:.1f}".format(acc_test)
            print(f'epoch: {epoch}\t err:{error_train}\t acc_train: {acc} \t acc_test: {acc_test} ')
            if epoch % 100==0:
                show_plot(X_test[start:end], a4, Y_test[start:end])

def show_plot(a1, a4, Y):
    nrows, ncols = 4, 5   # array of sub-plots
    figsize = [5, 5]  # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        pixels = a1[i].reshape((28, 28))
        axi.imshow(pixels)
        # write row/col indices as axes' title for identification
        axi.set_title("predict:" + str(np.argmax(a4[i])) + ", real:" + str(np.argmax(Y[i])))
        if np.argmax(a4[i])!= np.argmax(Y[i]):
            axi.plot('!!!!', color='red', linewidth=10)

    plt.tight_layout(True)
    plt.show()

def feedforward():
    pass

def backprop():
    pass

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data('./mnist_train.csv', "./mnist_test.csv")
    preprocess_data()
    init_neural_network(  X_train, Y_train, X_test, Y_test)
    feedforward()