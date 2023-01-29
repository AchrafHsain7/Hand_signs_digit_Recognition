import numpy as np
import matplotlib.pyplot as plt




def load_results():
        fileN = "data\parameters.npz" 
        data = np.load(fileN)
        W1 = data["W1"]
        b1 = data["b1"]
        W2 = data["W2"]
        b2 = data["b2"]
        W3 = data["W3"]
        b3 = data["b3"]

        parameters = {"W1":W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

        return parameters

def plot_cost_graph(model):
    # Plot learning curve (with costs)
    costs = np.squeeze(model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(model["learning_rate"]))
    plt.show()









        




        





        

