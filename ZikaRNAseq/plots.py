## plotting utils; original author: Zichen Wang (2014)
COLORS10 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def enlarge_tick_fontsize(ax, fontsize):
    ax.tick_params(axis='both', labelsize=fontsize)
