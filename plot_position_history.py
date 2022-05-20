import matplotlib.pyplot as plt
import numpy as np

formation = np.load('data/formation.npy')
formation -= np.reshape(np.mean(formation,axis=1), (1,1,2))
plt.scatter(formation[0,:,0], formation[0,:,1])

initial_position = np.load('data/init_pos.npy')
initial_position -= np.reshape(np.mean(initial_position,axis=1), (1,1,2))
plt.scatter(initial_position[0,:,0], initial_position[0,:,1])
print(initial_position.shape)

position_history = np.load('data/pos_history.npy')
position_history -= np.reshape(np.mean(position_history,axis=2),(position_history.shape[0], 1, 1, 2))

for i in range(3):
    plt.plot(position_history[:,0,i,0], position_history[:,0,i,1])

    plt.scatter(position_history[-1,0,i,0], position_history[-1,0,i,1],c='red')

plt.show()