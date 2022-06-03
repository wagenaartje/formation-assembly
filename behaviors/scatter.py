import matplotlib.pyplot as plt
import numpy as np

behaviors = np.load('./behaviors.npy')
fitness = np.load('./fitness.npy')

behaviors = behaviors[fitness[:,0] > -0.1,:]

plt.scatter(behaviors[:,0], behaviors[:,1])

plt.xlim(0,np.sqrt(2))
plt.ylim(0,1)
plt.gca().invert_yaxis()  # Makes more sense if larger velocities are on top.
plt.show()