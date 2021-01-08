import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import ipdb

# x = [1, 2, 3, 4]
x = np.arange(1,11)
# y = [4.3, 2.5, 3.5, 4.5]
# y = np.random.rand(100)
y = np.arange(1,11)
# label = [0,1,1,1,1,1,1,0]

cmap = ListedColormap(['b', 'r'])

points = np.array([x, y]).T.reshape(-1,1,2)
# segments = np.array([])
# for i in np.arange(len(y)):
    # segments = np.append(segments, np.concatenate([points[i+1],points[i+2]],axis=1))
#     segments = segments.reshape(-1,2,2)
#     ipdb.set_trace()
segments = np.concatenate([points[:-1], points[1:]], axis=1)
# np.concatenate[[points[i],points[i+1]]
# lc = LineCollection(segments, cmap=cmap)
label = np.where(y >= 5,1,0 )
# lc = LineCollection(points, cmap=cmap)


lc.set_array(np.array(label))
# ipdb.set_trace()
ax =  plt.subplot()
ax.add_collection(lc)
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.grid(True)
ipdb.set_trace()
plt.show()