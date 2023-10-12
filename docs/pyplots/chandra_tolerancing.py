from marxs.design.tolerancing import DispersedWigglePlotter

# In this example, we use a file that's included in MARXS for test purposes
from astropy.utils.data import get_pkg_data_filename
filename = get_pkg_data_filename('data/wiggle_global.fits', 'marxs.design.tests')

wiggle_plotter = DispersedWigglePlotter()
tab, fig = wiggle_plotter.load_and_plot(filename, ['dx', 'dy', 'dz', 'rx', 'ry', 'rz'])

fig.axes[0].legend()

for i in range(6):
    fig.axes[2 * i].set_ylim([0, 1e3])
    fig.axes[2 * i + 1].set_ylim([0, 70])

fig.set_size_inches(8, 6)
fig.subplots_adjust(left=.09, right=.93, top=.95, wspace=1.)
