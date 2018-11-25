from marxs.design.tolerancing import load_and_plot

tab, fig = load_and_plot('pyplots/wiggle_global.fits', ['dx', 'dy', 'dz', 'rx', 'ry', 'rz'])

fig.axes[0].legend()

for i in range(6):
    fig.axes[2 * i].set_ylim([0, 1e3])
    fig.axes[2 * i + 1].set_ylim([0, 70])

fig.set_size_inches(8, 6)
fig.subplots_adjust(left=.09, right=.93, top=.95, wspace=1.)
