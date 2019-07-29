import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import random

#Flower Data [Petal Length, Petal Width, Color(0 = Blue, 1 = Red)]
dataB1 = [1, 1, 0]
dataB2 = [2, 1, 0]
dataB3 = [2, .5, 0]
dataB4 = [3, 1, 0]

dataR1 = [3, 1.5, 1]
dataR2 = [3.5, .5, 1]
dataR3 = [4, 1.5, 1]
dataR4 = [5.5, 1, 1]

dataU = [4.5, 1, 'it should be 1']

allPoints = [dataB1, dataB2, dataB3, dataB4,
             dataR1, dataR2, dataR3, dataR4]

#set initial weights and bias
w1, w2, b = random.random(), random.random(), random.random()

#Set resolution of colormap (lower is higher)
dx, dy = 0.005, 0.005

#Set bounds of colormap
y, x = np.mgrid[slice(0, 3 + dy, dy),
                slice(0, 6 + dx, dx)]

#set z min and max for colormap construction
z_min, z_max = 0, 1

def NN(m1, m2, w1, w2, b):
    '''
    values : m1, m2 
    weights : w1, w2 
    bias : b 

    returns sigmoid of weighted value + bias 
    '''
    z = m1 * w1 + m2 * w2 + b 
    return sigmoid(z)

def sigmoid(x):
    # normalizes weighted value to be bounded by 0 and 1
    return 1/(1 + np.exp(-x))

def cost(pred, target):
    # squared difference cost function
    cost = (pred - target) ** 2
    return cost


def train(w1, w2, b):
    # training rate, dictates weight given to adjustments
    rate = 0.1
    
    #Pick a random point 
    tPoint = random.choice(allPoints)
    target = tPoint[2]

    #weighted value
    z = tPoint[0] * w1 + tPoint[1] * w2 + b

    #normalized value
    pred = NN(tPoint[0], tPoint[1], w1, w2, b)
    #derivative of sigmoid
    dpred = sigmoid(z) * (1 - sigmoid(z))

    cost = (pred - target) ** 2 
    #derivative of cost
    dcost = 2 * (pred - target)

    #Derivate of z with respect to w1
    dz_dw1 = tPoint[0]
    #Derivate of z with repect to w2
    dz_dw2 = tPoint[1]
    #Derivate of z with repect to b
    dz_db = 1 

    #adjustments to weights and bias
    dcost_w1 = dpred * dcost * dz_dw1
    dcost_w2 = dpred * dcost * dz_dw2
    dcost_b  = dpred * dcost * dz_db

    
    w1 -= dcost_w1 * rate 
    w2 -= dcost_w2 * rate 
    b  -= dcost_b * rate

    return w1, w2, b
    



def animate(i):
    
    #allow weights and bias to be modified
    global w1, w2, b

    #calulate total cost before modifications and save weights and bias
    totalCost = 0 
    sw1, sw2, sb = w1, w2, b
    for point in allPoints:
        totalCost += (sigmoid(NN(point[0], point[1], w1, w2 ,b)) - point[2]) ** 2

    #make adjustments to weights and bias with i iterations
    for i in range(1000):
        w1, w2, b = train(w1, w2, b)

    #calculate total cost after modifications
    ntotalCost = 0 
    for point in allPoints:
        ntotalCost += (sigmoid(NN(point[0], point[1], w1, w2, b)) - point[2]) ** 2

    #if the changes were detrimental to the model revert 
    if ntotalCost > totalCost:
        w1, w2, b = sw1, sw2, sb

    #update colormap
    graph = ax.pcolormesh(x, y, NN(x, y, w1, w2, b)[:-1, :-1], cmap='RdBu_r', vmin=z_min, vmax=z_max, zorder=1)

    return graph


Bx, By = [], []
Rx, Ry = [], []
fig, ax = plt.subplots()

for point in allPoints:
    if point[2] == 0:
        Bx.append(point[0])
        By.append(point[1])
    elif point[2] == 1:
        Rx.append(point[0])
        Ry.append(point[1])


graph = ax.pcolormesh(x, y, NN(x, y, w1, w2, b)[:-1, :-1], cmap='RdBu_r', vmin=z_min, vmax=z_max, zorder=1)

plt.title('Neural Network Visualization: 10k iterations per update')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

cbar = plt.colorbar(graph)
cbar.ax.set_ylabel('Flower Guess')

ax.scatter(Bx, By, c='b', label='Blue Flowers', zorder=4)
ax.scatter(Rx, Ry, c='r', label = 'Red Flowers', zorder=3)
ax.scatter(dataU[0], dataU[1], c='k', label='Unknown Flower', zorder=5)

ax.legend()

plt.tight_layout()

anim = FuncAnimation(fig, animate, frames = 100, interval = 200)

plt.show()