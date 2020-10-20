import os
import pickle
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.stats import mode, iqr
import matplotlib.colors as clr
from matplotlib import cm
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

plt.rcParams["font.family"] = 'Serif'
plt.rcParams["legend.labelspacing"] = 3.4
plt.rcParams["legend.handletextpad"] = -1
plt.rcParams["figure.titlesize"] = 'xx-large'
plt.rcParams["axes.labelsize"] = 'x-large'
plt.rcParams["ytick.major.size"] = 10
plt.rcParams["xtick.major.size"] = 10

colors = ['#a171d4', '#7ed14d', '#79c5da', '#fd787b']
lighter_colors = ['#e2d4f2', '#d8f1c9', '#d6edf3', '#fed6d7']
sns.set_palette(sns.color_palette(colors))


class HandlerColormap(HandlerBase):
    def __init__(self, cmap, num_stripes=8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize,
                       trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle([ydescent, xdescent - 5 + i * 60 / self.num_stripes],
                          8,
                          60 / self.num_stripes,
                          fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                          transform=trans)
            stripes.append(s)
        return stripes


# # Load dataset
nc = netCDF4.Dataset('./data/cam5.1_amip_1d_002.cam2.h1.PRECL.19790101-20051231.nc', 'r')
lats = nc.variables['lat'][:]
lons = nc.variables['lon'][:]

seed = 8
order = 4

save_folder = 'output'
save_file = os.path.join(save_folder,
                         'gmr_aggregation_case_' + str(seed) + 'order' + str(order) + '.pickle')
with open(save_file, 'rb') as f:
    output = pickle.load(f)

cluster = output['cluster']

with open('./prec_mask.pickle', 'rb') as f:
    masks = pickle.load(f)
masks = np.stack(masks)
"""
For each location, record:
1) the number of prec days
2) the cluster
"""
wet_days = np.zeros((lats.shape[0], lons.shape[0]))
cluster_mode = np.zeros((lats.shape[0], lons.shape[0]))

for i, _ in enumerate(lats):
    for j, _ in enumerate(lons):
        slices = np.logical_and(masks[:, 1] == i, masks[:, 2] == j)
        clusters = cluster[slices]
        if clusters.shape[0] == 0:
            cluster_mode[i, j] = 0
            wet_days[i, j] = 0
        else:
            cluster_mode[i, j] = int(mode(cluster[slices])[0][0])
            wet_days[i, j] += slices.sum()
        # print(i, j, cluster_mode[i,j], wet_days[i, j], clusters.shape[0])

np.save('gmr_cluster' + 'seed_' + str(seed) + 'order_' + str(order) + '.npy', cluster_mode)
np.save('gmr_wet_days' + 'seed_' + str(seed) + 'order_' + str(order) + '.npy', wet_days)
"""
plot by the number of wet days
color by the clusters

"""

# precipitation, Q, T, OMEGA grouped by clusters

PRECL = np.load('PRECL.npy')
OMEGA = np.load('OMEGA.npy')
Q = np.load('Q.npy')
T = np.load('T.npy')

# seed = 7
# order = 6
save_folder = 'output'
save_file = os.path.join(save_folder,
                         'gmr_aggregation_case_' + str(seed) + 'order' + str(order) + '.pickle')
with open(save_file, 'rb') as f:
    output = pickle.load(f)

cluster = output['cluster']

df = pd.DataFrame({'cluster': cluster, 'PRECL': PRECL.squeeze()})
df['cluster'] = df['cluster'].astype('category')

sorted_index = df.groupby('cluster').median().sort_values('PRECL').index

# permute the cluster number in the original array
for i in range(order):
    cluster[cluster == i] = np.where(sorted_index.values == i)[0][0] + order

cluster = cluster - order

df = pd.DataFrame({'cluster': cluster + 1, 'PRECL': PRECL.squeeze()})
df['cluster'] = df['cluster'].astype('category')

fig, ax = plt.subplots(figsize=(5, 5))
g = sns.boxplot(x='cluster',
                y='PRECL',
                data=df,
                flierprops=dict(markerfacecolor='0.50', markersize=0.1),
                ax=ax)
g.set_yscale('log')

plt.ylabel("PRECL (mm/day)")
plt.title('Precipitation', fontdict={'fontsize': 20})
plt.savefig('seed_' + str(seed) + '_order' + str(order) + '_gmr_reduced_PRECL_by_cluster.png',
            dpi=100,
            bbox_inches='tight',
            pad_inches=0.05)

fig, ax = plt.subplots(figsize=(7, 15))
levels = np.arange(30)[::-1] - 1
for i in range(order):
    mean = np.median(Q[cluster == i], 0)
    quantiles = np.quantile(Q[cluster == i], [0.25, 0.75], 0)
    ax.plot(mean, levels, c=colors[i], linestyle='dashed', linewidth=3)
    ax.fill_betweenx(levels, quantiles[0], quantiles[1], color=colors[i], alpha=.5)
    plt.yticks(np.arange(0, 30, 2), [str(i + 1) for i in range(0, 30, 2)][::-1])
    plt.xlabel('rQ (kg $\text{H}_2$O/ kg air)')
    plt.ylabel('Altitude levels')

plt.grid(which='both', axis='y')
plt.title('Moisture Content', fontdict={'fontsize': 20})
plt.savefig('seed_' + str(seed) + '_order' + str(order) + '_gmr_reduced_Q_by_cluster.png',
            dpi=100,
            bbox_inches='tight',
            pad_inches=0.05)

fig, ax = plt.subplots(figsize=(7, 15))
levels = np.arange(30)[::-1] - 1
for i in range(order):
    mean = np.median(OMEGA[cluster == i], 0)
    quantiles = np.quantile(OMEGA[cluster == i], [0.25, 0.75], 0)
    # sd = iqr(OMEGA[cluster == i],0)
    ax.plot(mean, levels, c=colors[i], linestyle='dashed', linewidth=3)
    ax.fill_betweenx(levels, quantiles[0], quantiles[1], color=colors[i], alpha=.5)
    plt.yticks(np.arange(0, 30, 2), [str(i + 1) for i in range(0, 30, 2)][::-1])
    plt.xlabel('OMEGA (Pascal/s)')
    plt.ylabel('Altitude levels')
plt.grid(which='both', axis='y')
plt.title('Vertical Velocity', fontdict={'fontsize': 20})
plt.savefig('seed_' + str(seed) + '_order' + str(order) + '_gmr_reduced_OMEGA_by_cluster.png',
            dpi=100,
            bbox_inches='tight',
            pad_inches=0.05)

fig, ax = plt.subplots(figsize=(7, 15))
levels = np.arange(30)[::-1] - 1
for i in range(order):
    mean = np.median(T[cluster == i], 0)
    quantiles = np.quantile(T[cluster == i], [0.25, 0.75], 0)
    # sd = iqr(T[cluster == i],0)
    ax.plot(mean, levels, c=colors[i], linestyle='dashed', linewidth=3)
    ax.fill_betweenx(levels, quantiles[0], quantiles[1], color=colors[i], alpha=.5)
    plt.xlabel('T (Kelvin)')
    plt.ylabel('Altitude levels')
    plt.yticks(np.arange(0, 30, 2), [str(i + 1) for i in range(0, 30, 2)][::-1])

plt.grid(which='both', axis='y')
plt.title('Temperature', fontdict={'fontsize': 20})
plt.savefig('seed_' + str(seed) + '_order' + str(order) + '_gmr_reduced_T_by_cluster.png',
            dpi=100,
            bbox_inches='tight',
            pad_inches=0.05)

cluster = np.load('gmr_cluster' + 'seed_' + str(seed) + 'order_' + str(order) + '.npy')
wet_days = np.load('gmr_wet_days' + 'seed_' + str(seed) + 'order_' + str(order) + '.npy')

fig, ax = plt.subplots(figsize=(10, 5))

m = Basemap(
    llcrnrlon=lons.min(),  # The lower left corner geographical longitude
    llcrnrlat=lats.min(),  # The lower left corner geographical latitude
    urcrnrlon=lons.max(),  # The upper right corner geographical longitude
    urcrnrlat=lats.max(),  # The upper right corner geographical latitude
    resolution='c',
    projection='cyl',
    ax=ax)

parallels = np.arange(-80., 80, 30.)
m.drawparallels(parallels, labels=[True, False, True, False], fontsize=15)
meridians = np.arange(0., 360., 60.)
m.drawmeridians(meridians, labels=[True, False, False, True], fontsize=15)

cmaps = []
for i in range(order):
    lat_idx, lon_idx = np.where(cluster == i)
    cmap = clr.LinearSegmentedColormap.from_list('custom' + str(i), [lighter_colors[i], colors[i]],
                                                 N=15)
    cmaps.append(cmap)
    m.scatter(lons[lon_idx],
              lats[lat_idx],
              c=wet_days[cluster == i],
              s=3,
              vmin=wet_days[cluster == i].min(),
              vmax=wet_days[cluster == i].max(),
              marker='.',
              cmap=cmap)

# Add Coastlines, States, and Country Boundaries
m.drawcoastlines()
m.drawcountries()

cmap_labels = ["1", "2", "3", "4"]
# create proxy artists as handles:
cmap_handles = [Rectangle((0, 0), 0.1, 1) for _ in cmaps]
handler_map = dict(zip(cmap_handles, [HandlerColormap(cm, num_stripes=15) for cm in cmaps]))

leg = ax.legend(handles=cmap_handles,
                labels=cmap_labels,
                handler_map=handler_map,
                title="Cluster",
                fontsize='x-large',
                bbox_to_anchor=(1.01, 0., .1, 1),
                loc='lower left',
                borderaxespad=0.)

ax.add_artist(leg, )
plt.savefig('seed_' + str(seed) + '_order' + str(order) + '_gmr_reduced.png',
            dpi=100,
            bbox_inches='tight',
            pad_inches=0.05)
