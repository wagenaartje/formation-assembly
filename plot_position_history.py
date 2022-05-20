import matplotlib.pyplot as plt
import numpy as np

position_history = np.load('positions_save.npy')

for i in range(3):
    plt.plot(position_history[:,0,i,0], position_history[:,0,i,1])

    plt.scatter(position_history[-1,0,i,0], position_history[-1,0,i,1],c='red')



plt.show()