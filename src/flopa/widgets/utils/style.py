# flopa/widgets/utils/style.py

def dark_plot(ax, fig):
    """Applies a dark theme to a matplotlib axis and figure."""
    dark_grey = "#262930"  # Napari's dark grey 262930
    light_grey = "#E0E0E0"  # A light grey for text

    fig.set_facecolor(dark_grey)
    ax.set_facecolor(dark_grey)

    for spine in ax.spines.values():
        spine.set_color(light_grey)

    ax.xaxis.label.set_color(light_grey)
    ax.yaxis.label.set_color(light_grey)
    ax.title.set_color(light_grey)

    ax.tick_params(axis='x', colors=light_grey)
    ax.tick_params(axis='y', colors=light_grey)

    if ax.get_legend() is not None:
        legend = ax.get_legend()
        legend.get_frame().set_facecolor(dark_grey)
        legend.get_frame().set_edgecolor(light_grey)
        for text in legend.get_texts():
            text.set_color(light_grey)

def light_plot(ax, fig):
    """Applies a light theme to a matplotlib axis and figure."""
    light_bg = "#E0E0E0"      # Pure white background
    dark_text = "#262930"     # Black text for high contrast
    light_grid = "#E0E0E0"    # Light grey for grid lines and spines

    fig.set_facecolor(light_bg)
    ax.set_facecolor(light_bg)

    for spine in ax.spines.values():
        spine.set_color(dark_text)

    ax.xaxis.label.set_color(dark_text)
    ax.yaxis.label.set_color(dark_text)
    ax.title.set_color(dark_text)

    ax.tick_params(axis='x', colors=dark_text)
    ax.tick_params(axis='y', colors=dark_text)

    #ax.grid(True, color=light_grid, linestyle='--', linewidth=0.5)

    if ax.get_legend() is not None:
        legend = ax.get_legend()
        legend.get_frame().set_facecolor(light_bg)
        legend.get_frame().set_edgecolor(light_grid)
        for text in legend.get_texts():
            text.set_color(dark_text)
