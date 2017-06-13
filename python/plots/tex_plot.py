# initial import
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib import rcParams
from matplotlib import ticker

# create figure object with LaTeX font
fig = plt.figure(1)
matplotlib.rcParams.update({'font.size': 22, 'text.usetex': True})
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)


# plot something here
plt.plot(xx, samples, color='navy')
plt.xlabel('Number of samples')
plt.ylabel('Models $k$ (dimension)')
ttl = plt.title('Uniform prior on $k$',fontsize=24)
ttl.set_position([.5, 1.02])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


# save pdf image

plt.tight_layout()
plt.savefig('chain_evol.pdf', format='pdf', dpi=50, bbox_inches='tight')


#---------------------------------------------------------------------
# 3D Plot
min_x = min(X.flatten())
min_y = min(Y.flatten())
min_z = min(Z.flatten())

max_x = max(X.flatten())
max_y = max(Y.flatten())
max_z = max(Z.flatten())

fig = plt.figure()

ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z, rstride=5, cstride=5, cmap=cm.pink_r, antialiased=False)
ax.plot_wireframe(X,Y,Z, rstride=5, cstride=5, linewidth=0.5, color='black')
ax.view_init(elev=30, azim=-130)

# axes and title config
ax.set_xlabel('$x$', labelpad=15)
ax.yaxis.set_rotate_label(False) # disable automatic rotation
ax.set_ylabel('$x^\prime$', rotation = 0, labelpad=15)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('$C_{HH}(x,x^\prime)$',rotation=93, labelpad=3)
ax.set_xlim3d(min_x, max_x)
ax.set_ylim3d(min_y, max_y)
ax.set_zlim3d(min_z, max_z)
ttl = ax.set_title('Nice title')
ttl.set_position([.5, 0.95])

plt.tight_layout()