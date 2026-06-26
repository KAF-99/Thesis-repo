"""Shared thesis figure style: one Okabe-Ito, colourblind-safe, vector-PDF theme for every chapter."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe qualitative palette (unchanged from Chapter 5).
OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
             '#CC79A7', '#56B4E9', '#F0E442', '#000000']

# Named mapping from the five analysis regimes to colours, so every chapter
# shades the same regimes identically (matches the Chapter-5 regime figure).
REGIME_COLORS = {
    'Post-GFC / Euro-crisis': OKABE_ITO[0],
    'Low-rate I':             OKABE_ITO[1],
    'Low-rate II':            OKABE_ITO[2],
    'COVID':                  OKABE_ITO[3],
    'Hiking':                 OKABE_ITO[4],
}

# Semantic constants used across figures.
SERIES_COLOR = '#000000'   # near-black line colour for the main series
FIG_WIDTH = 11             # default figure width (inches)


def fig_size(height):
    """Return (FIG_WIDTH, height) for a standard-width figure."""
    return (FIG_WIDTH, height)


def apply_thesis_style():
    """Apply the shared publication rcParams (vector PDF, despined, Okabe-Ito cycle)."""
    mpl.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'ps.fonttype': 42,            # editable text in vector PDF
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.prop_cycle': mpl.cycler(color=OKABE_ITO),
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
        'legend.frameon': False, 'figure.autolayout': False,
    })


def save_fig(fig, name, caption, fig_dir):
    """Save a vector PDF to ``fig_dir`` and write its one-paragraph caption .txt.

    Generalised from the linear notebook's inline ``save_fig`` (which closed over
    a notebook ``FIG_DIR`` global) to take ``fig_dir`` as an explicit argument;
    behaviour is otherwise identical.
    """
    path = f'{fig_dir}/{name}.pdf'
    fig.savefig(path, bbox_inches='tight')
    with open(f'{fig_dir}/{name}.txt', 'w') as fh:
        fh.write(caption.strip() + '\n')
    plt.close(fig)
    print(f'  saved figure: {path}  (+ caption .txt)')
