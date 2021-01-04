# # #!/usr/bin/env python
# # '''
# # Color parts of a line based on its properties, e.g., slope.
# # '''
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from matplotlib.collections import LineCollection
# # from matplotlib.colors import ListedColormap, BoundaryNorm
# # import ipdb

# # x = np.linspace(0, 3 * np.pi, 500)
# # y = np.sin(x)
# # z = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

# # # Create a colormap for red, green and blue and a norm to color
# # # f' < -0.5 red, f' > 0.5 blue, and the rest green
# # cmap = ListedColormap(['r','b'])
# # norm = BoundaryNorm([-1,-0.5,0.5,1], cmap.N)

# # # Create a set of line segments so that we can color them individually
# # # This creates the points as a N x 1 x 2 array so that we can stack points
# # # together easily to get the segments. The segments array for line collection
# # # needs to be numlines x points per line x 2 (x and y)
# # points = np.array([x, y]).T.reshape(-1,1,2)
# # segments = np.concatenate([points[:-1], points[1:]], axis=1)
# # ipdb.set_trace()

# # # Create the line collection object, setting the colormapping parameters.
# # # Have to set the actual values used for colormapping separately.
# # lc = LineCollection(segments, cmap=cmap, norm=norm)
# # lc.set_array(z)
# # plt.gca().add_collection(lc)

# # plt.xlim(x.min(), x.max())
# # plt.ylim(-1.1, 1.1)
# # plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
# import ipdb

# x = np.linspace(0, 3 * np.pi, 500)
# y = np.sin(x)
# dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
# ipdb.set_trace()
# # Create a set of line segments so that we can color them individually
# # This creates the points as a N x 1 x 2 array so that we can stack points
# # together easily to get the segments. The segments array for line collection
# # needs to be (numlines) x (points per line) x 2 (for x and y)
# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)

# fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

# # Create a continuous norm to map from data points to colors
# norm = plt.Normalize(dydx.min(), dydx.max())
# lc = LineCollection(segments, cmap='viridis', norm=norm)
# # Set the values used for colormapping
# lc.set_array(dydx)
# lc.set_linewidth(2)
# line = axs[0].add_collection(lc)
# fig.colorbar(line, ax=axs[0])

# # Use a boundary norm instead
# cmap = ListedColormap(['r', 'g', 'b'])
# norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
# lc = LineCollection(segments, cmap=cmap, norm=norm)
# lc.set_array(dydx)
# lc.set_linewidth(2)
# line = axs[1].add_collection(lc)
# fig.colorbar(line, ax=axs[1])

# axs[0].set_xlim(x.min(), x.max())
# axs[0].set_ylim(-1.1, 1.1)
# plt.show()

import matplotlib.pyplot as plt
import math
import matplotlib
import ipdb

N=128
f = 4
# N = 8
omg = [i*f*2*math.pi/N for i in range(N)]
sig = [math.sin(i) for i in omg]
ipdb.set_trace()

fig,(ax) = plt.subplots(1,1)

for j in range(len(sig)-1):
    ax.plot(omg[j:j+2], sig[j:j+2])

# plt.show()
RGBA=[1,0,0,1]
ax.set_color_cycle(RGBA)
N=128
f = 8
omg = [i*f*2*math.pi/N for i in range(N)]
sig = [math.sin(i) for i in omg]

fig,(ax) = plt.subplots(1,1)
colormap = plt.get_cmap('jet')

ax.set_color_cycle([colormap(k) for k in sig])

for j in range(len(sig)-1):
    ax.plot(omg[j:j+2], sig[j:j+2])

plt.show()