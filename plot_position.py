import numpy as np
import matplotlib.pyplot as plt

position = np.load('best_position.npy')
initial_position = np.load('initial_position.npy')

formation = np.asarray([[0,0], [1/np.sqrt(2), 1/np.sqrt(2)], [2/np.sqrt(2),0]])
formation = np.reshape(formation, (1,3,2))
formation -= np.reshape(np.mean(formation,axis=1),(1,1,2))

plt.scatter(position[:,0], position[:,1], label='actual')
plt.scatter(formation[0,:,0], formation[0,:,1], label='target')
plt.scatter(initial_position[:,0], initial_position[:,1], label='start')

print(formation[0,:,0])
plt.legend()
plt.show()