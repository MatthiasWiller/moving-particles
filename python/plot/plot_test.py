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