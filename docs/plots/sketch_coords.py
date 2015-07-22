"""This module make plots and sketches for the documentation."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def optical_element_coords():
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.quiver([2,0,0],[0,2,0],[0,0, 2],[1,0,0],[0,1,0], [0,0,1], length=4, color='k', lw=2)
   ax.bar3d(-1, -1, -1, 1, 2, 2, zsort='average', alpha=0.5, color=['b'] *5 + ['r'], zorder=1)
   ax.set_axis_off()
   ax.text(2.1, 0, 0, 'x', None, fontsize='x-large')
   ax.text(0, 2.1, 0, 'y', None, fontsize='x-large')
   ax.text(0, 0, 2.1, 'z', None, fontsize='x-large')
   ax.set_xlim([-1.2,1.2])
   ax.set_ylim([-1.2,1.2])
   ax.set_zlim([-1.2,1.2])
   fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
   plt.show()
