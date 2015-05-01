__author__ = 'omoore'

import matplotlib.pyplot as plt
import matplotlib.colors as cls
import numpy as np
from pylab import savefig
import colorPicker


def show_colormap(colormaps):
    im = np.outer(np.ones(5), np.arange(100))
    fig, ax = plt.subplots(len(colormaps), figsize=(6, len(colormaps)/2), subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.1)
    for idx, (title, cmap) in enumerate(sorted(colormaps.items())):
        ax[idx].imshow(im, cmap=cmap)
        ax[idx].set_title(title)
    plt.tight_layout()
    savefig('ui_files/images/colorMaps.png', dpi=200)

    plt.show()



color_maps = colorPicker.retrieve_colormaps(reversed=False)[1]
show_colormap(color_maps)


