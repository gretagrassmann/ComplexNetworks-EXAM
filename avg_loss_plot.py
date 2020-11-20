import matplotlib.pyplot as plt
import numpy as np
relu = "C:\\Users\\Cobal\\Desktop\\relu.txt"
theta = "C:\\Users\\Cobal\\Desktop\\ComplexNetworksEXAM\\ComplexNetworks2\\Sigmoidal_avg_loss_train.txt"

# Import data as a list of numbers
with open(relu) as textFile:
    data = textFile.read().split()          # split based on spaces
    data = [float(point) for point in data] # convert strings to floats


with open(theta) as textFile:
    data1 = textFile.read().split()          # split based on spaces
    data1 = [float(point) for point in data1] # convert strings to floats

x = np.arange(1,len(data)+1)

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x, data, label='With ReLU as activation function')
ax.plot(x, data1, label='With $\\theta$ as activation function')
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Average loss')
#plt.title('Comparison ')
ax.legend()
plt.show()
