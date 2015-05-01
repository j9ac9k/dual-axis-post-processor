from matplotlib import cm

__author__ = 'omoore'
import seaborn as sns
import matplotlib.colors as cls
import matplotlib.pyplot as plt
def retrieve_colormaps(color_name='blue', reversed=False):
    seq_suffix = '_d'
    if reversed:
        seq_suffix = ''

    seq_color_name_dict = {'red': 'Reds' + seq_suffix,
                           'blue': 'Blues' + seq_suffix,
                           'green': 'Greens' + seq_suffix,
                           'purple': 'Purples' + seq_suffix,
                           'orange': 'Oranges' + seq_suffix,
                           'blue-green': 'BuGn' + seq_suffix,
                           'blue-purple': 'BuPu' + seq_suffix,
                           'green-blue': 'GnBu' + seq_suffix,
                           'orange-red': 'OrRd' + seq_suffix,
                           'purple-blue': 'PuBu' + seq_suffix,
                           'purple-blue-green': 'PuBuGn' + seq_suffix,
                           'purple-red': 'PuRd' + seq_suffix,
                           'red-purple': 'RdPu' + seq_suffix,
                           'yellow-green': 'YlGn' + seq_suffix,
                           'yellow-green-blue': 'YlGnBu' + seq_suffix,
                           'yellow-orange-brown': 'YlOrBr' + seq_suffix,
                           'yellow-orange-red': 'YlOrRd' + seq_suffix,
                           'cube-helix': 'cubehelix' + seq_suffix}

    std_suffix = ''
    if reversed:
        std_suffix = '_r'
    std_color_name_dict = {'flag': 'flag' + std_suffix,
                           'jet': 'jet' + std_suffix}

    #available_color_maps = [None] * len(color_name_dict)
    available_color_maps = dict.fromkeys(seq_color_name_dict.keys())

    for color in seq_color_name_dict.keys():
        available_color_maps[color] = cls.ListedColormap(sns.color_palette(seq_color_name_dict[color], 256))

    for color in std_color_name_dict.keys():
        available_color_maps[color] = plt.get_cmap(std_color_name_dict[color])


    cmap = available_color_maps[color_name]
    return cmap, available_color_maps
