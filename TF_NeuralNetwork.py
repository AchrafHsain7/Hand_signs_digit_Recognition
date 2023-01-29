import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import matplotlib.pyplot as plt
import numpy as np
import h5py

#function for normalizing the data
def normalize(image):
  
  image = tf.cast(image, dtype= tf.float32) / 255.
  image = tf.reshape(image, [-1, ])
  return image

#creating a one hot function to transform Y into a more convenient matrix
def one_hot_matrix(labels, depth=6):
  one_hot = tf.reshape(tf.one_hot(labels, depth, axis=0), shape=[-1,])
  return one_hot

def initialize_parameters():

  #creating the initializer Gorots will be used here
  initializer = tf.keras.initializers.GlorotNormal()

  #creating the parameters
  W1 = tf.Variable(initializer(shape=(25, 12288)))
  b1 = tf.Variable(initializer(shape=(25, 1)))
  W2 = tf.Variable(initializer(shape=(12, 25)))
  b2 = tf.Variable(initializer(shape=(12, 1)))
  W3 = tf.Variable(initializer(shape=(6, 12)))
  b3 = tf.Variable(initializer(shape=(6, 1)))

  parameters = {
      'W1': W1,
      'b1': b1,
      'W2': W2,
      'b2': b2,
      'W3': W3,
      'b3': b3
  }

  return parameters

#creating forward propagation function
def forward_propagation(X, parameters):

  #unpacking the parameters
  W1 = parameters['W1']
  b1 = parameters['b1']
  W2 = parameters['W2']
  b2 = parameters['b2']
  W3 = parameters['W3']
  b3 = parameters['b3']

  #forward propagation using reLu 
  Z1 = tf.add(tf.linalg.matmul(W1, X) , b1)
  A1 = tf.keras.activations.relu(Z1)
  Z2 = tf.add(tf.linalg.matmul(W2, A1), b2)
  A2 = tf.keras.activations.relu(Z2)
  Z3 = tf.add(tf.linalg.matmul(W3, A2), b3)

  return Z3

def calculate_loss(logits, labels):
  #logits are the results we got in the prediction process
  #labels are the results we should get

  loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=True))

  return loss



def model(X, Y, X_testing, Y_testing, learning_rate=0.0001, num_epochs=1500, mini_batch_size=32, print_cost=True):
  #this model function will train and test the resukt of the trained model to then return the needed data for storage

  #creating lists to keep track of the cost and accuracy's
  costs = []
  train_acc = []
  test_acc = []

  #initialize parameters
  parameters = initialize_parameters()

  #unpacking the parameters
  W1 = parameters['W1']
  b1 = parameters['b1']
  W2 = parameters['W2']
  b2 = parameters['b2']
  W3 = parameters['W3']
  b3 = parameters['b3']

  #generating our optimizer (Adam in this case)
  optimizer = tf.keras.optimizers.Adam(learning_rate)

  #for tracking the accuracy use categoricalAccuracy
  train_accuracy = tf.keras.metrics.CategoricalAccuracy()
  test_accuracy = tf.keras.metrics.CategoricalAccuracy()

  #creating the 'dataset' aka mapping X with Y
  dataset = tf.data.Dataset.zip((X, Y))
  test_dataset = tf.data.Dataset.zip((X_testing, Y_testing))

  #getting the number of training examples 
  m = dataset.cardinality().numpy()

  #creating the mini_batches
  mini_batches = dataset.batch(mini_batch_size).prefetch(8)
  test_mini_batches = test_dataset.batch(mini_batch_size).prefetch(8)

  #training the neural Network
  for epoch in range(num_epochs):
    epoch_total_loss = 0

    #iterating over the minibatches
    for (mini_batch_X, mini_batch_Y) in mini_batches:

      with tf.GradientTape() as tape:
        Z3 = forward_propagation(tf.transpose(mini_batch_X), parameters)
        mini_batch_loss = calculate_loss(Z3, tf.transpose(mini_batch_Y))

      #getting the accuracy of the current mini_batch
      train_accuracy.update_state(mini_batch_Y, tf.transpose(Z3))

      #getting the gradients by iterating over the variables 
      trainable_variables = [W1, b1, W2, b2, W3, b3]
      grads = tape.gradient(mini_batch_loss, trainable_variables)
      #applying adam optimization
      optimizer.apply_gradients(zip(grads, trainable_variables))
      epoch_total_loss += mini_batch_loss

    epoch_total_loss /= m

    #printing the cost every 100 epoch
    if print_cost and (epoch%10 == 0 or epoch == num_epochs-1):
      print('Cost at epoch ' + str(epoch) +': ' + str(epoch_total_loss.numpy()))
      print('Train Accuracy: ', train_accuracy.result().numpy())

      #updating the testing accuracy every 10 epochs for less compuattations later
      for (mini_batch_X, mini_batch_Y) in test_mini_batches:
        Z3 = forward_propagation(tf.transpose(mini_batch_X), parameters)
        test_accuracy.update_state(mini_batch_Y, tf.transpose(Z3))
      print("Test Accuracy: " + str(test_accuracy.result().numpy()) + '\n')

      #updating the lists for later data
      costs.append(epoch_total_loss)
      train_acc.append(train_accuracy.result())
      test_acc.append(test_accuracy.result())
      #resering the state of test accuracy
      test_accuracy.reset_state()
  
  #returning the results and data
  return parameters, costs, train_acc, test_acc

#storing the parameters with numpy
def save_results_tf( parameters):
        W1 = parameters["W1"].numpy()
        b1 = parameters["b1"].numpy()
        W2 = parameters["W2"].numpy()
        b2 = parameters["b2"].numpy()
        W3 = parameters["W3"].numpy()
        b3 = parameters["b3"].numpy()

        np.savez('parameters', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


def main():
    #loading the data from the dataset
    train_dataset = h5py.File("train_signs.h5", 'r')
    test_dataset = h5py.File("test_signs.h5", 'r')



    #slicing it using tenserflow int X, Y
    new_X_train = tf.data.Dataset.from_tensor_slices(train_dataset["train_set_x"])
    new_Y_train = tf.data.Dataset.from_tensor_slices(train_dataset["train_set_y"])



    new_X_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    new_Y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])



    labels = set()
    for element in new_Y_train:
        labels.add(element.numpy())


    #normalizing the data with map fct
    X_train = new_X_train.map(normalize)
    X_test = new_X_test.map(normalize) 
    Y_train = new_Y_train.map(one_hot_matrix)
    Y_test = new_Y_test.map(one_hot_matrix)

    parameters,costs ,train_acc ,test_acc = model(X_train, Y_train, X_test, Y_test)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(0.0001))
    plt.show()

    # Plot the train accuracy
    plt.plot(np.squeeze(train_acc))
    plt.ylabel('Train Accuracy')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(0.0001))

    # Plot the test accuracy
    plt.plot(np.squeeze(test_acc))
    plt.ylabel('Test Accuracy')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(0.0001))
    plt.show()

    save_results_tf(parameters)