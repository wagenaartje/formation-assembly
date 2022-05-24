import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

archive = np.load('archive.npy')
chunk = int(archive.shape[0]/10)

cmap = matplotlib.cm.get_cmap('jet')

#for i in range(10):
#    plt.scatter(archive[chunk*i:chunk*(i+1),0], archive[chunk*i:chunk*(i+1),1],10,label=str(i),color=cmap(i/9))

colors = cmap(np.arange(archive.shape[0])/archive.shape[0])
plt.scatter(archive[:,0], archive[:,1],10,c=colors)

plt.xlabel('Minimum dist')
plt.ylabel('Time')
plt.legend()

plt.show()