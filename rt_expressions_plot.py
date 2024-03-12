import matplotlib.pyplot as plt 
import numpy as np

def plot_training(loss_data, accuracy_data, bs, e, lr, eval, name):
    x = np.arange(0, len(loss_data)) 
  
    # plot lines 
    plt.title(f"bs = {bs}, e = {e}, lr = {lr}, eval = {eval:.2f}%")

    plt.plot(x, accuracy_data, label = f"Accuracy | max: {max(accuracy_data):.1f}%")
    plt.plot(x, loss_data, label = f"Loss | min: {min(loss_data):.3f}")
    plt.legend() 
    plt.savefig("diagrams/"+name+".png")
    