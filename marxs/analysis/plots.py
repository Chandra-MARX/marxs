# Licensed under GPL version 3 - see LICENSE.rst

import numpy as np

class OrderColor():
    '''Consistent color scheme for plots that show multiple different orders

    Parameters
    ---------
    colormap : string
        Name of a matplotlib colormap
    max_order : int
        Maximum order that this color scheme will be used on (i.e. the end of
        the colorbar will be assigned ot this order)
    '''
    def __init__(self, colormap='nipy_spectral', max_order=15):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        self.color = cmap(np.linspace(0, 1, abs(max_order + 1)))

    def __call__(self, order):
        kwargs = {}
        if order == 0:
            kwargs['color'] = 'k'
        else:
            kwargs['color'] = self.color[abs(order)]
        if order <= 0:
            kwargs['linestyle'] = '-'
        if order > 0:
            kwargs['linestyle'] = ':'
        if order == 0:
            kwargs['lw'] = 4
            kwargs['alpha'] = .7
        else:
            kwargs['lw'] = 2
        return kwargs
