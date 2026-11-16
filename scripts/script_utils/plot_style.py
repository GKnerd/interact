"""Figure style for papers and reports, shared by the evaluation scripts.

- Axis labels as quantity / unit (ISO 80000-1), e.g. '$t$ / s'. Quantity symbols italic,
  units, descriptive subscripts and operators upright (ISO 80000-2), e.g. '$e_\\mathrm{FDE}$ / mm'.
- Serif text (STIX, Times-like, shipped with matplotlib) at 8 pt, the body size of IEEE papers.
- Fonts embedded as TrueType (Type 42), as IEEE PDF eXpress requires; vector PDF output.
- One fixed colour per method in every figure. Line style and marker repeat the identity,
  so the figures stay readable in grayscale print and for colour-vision deficiencies.

Lives in scripts/script_utils/, not in the package: importing interact never loads matplotlib.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4                  # inch per mm
COLUMN_WIDTH = 88.9 * MM       # IEEE single column
TEXT_WIDTH = 181.0 * MM        # IEEE double column

INK = '#0b0b0b'                # text, ground truth
INK_SECONDARY = '#52514e'
MUTED = '#898781'              # axes, ticks, context series (history, training curve)
GRID = '#e1e0d9'

# Categorical slots 1-3 of the reference palette, validated as a set (all pairs, incl. CVD)
METHOD_STYLE = {
    'model': dict(color='#2a78d6', linestyle='-', marker='o', label='Model'),
    'zero_velocity': dict(color='#eb6834', linestyle=(0, (4, 1.5)), marker='s',
                          label='Zero velocity'),
    'constant_velocity': dict(color='#1baf7a', linestyle=(0, (1, 1.2)), marker='^',
                              label='Constant velocity'),
}

GROUP_LABEL = {'all': 'All joints', 'wrists_hands': 'Wrists and hands'}

# Axis labels: quantity / unit (ISO 80000-1)
LABEL_TIME = 'Forecast time $t$ / s'
LABEL_MPJPE = r'$e_\mathrm{MPJPE}$ / mm'
LABEL_FDE = r'$e_\mathrm{FDE}$ / mm'
LABEL_ERROR = r'Joint position error $e$ / mm'
LABEL_SPEED = r'Mean joint speed in history $\bar{v}$ / (m$\,$s$^{-1}$)'
LABEL_SHARE = 'Share of windows / %'
LABEL_HEADING = r'Heading error $\Delta\psi$ / °'
LABEL_EPOCH = 'Epoch'


def apply_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.unicode_minus': True,          # a real minus sign, not a hyphen
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.color': INK,
        'axes.labelcolor': INK,
        'axes.titlecolor': INK,
        'axes.edgecolor': MUTED,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.4,
        'grid.linestyle': '-',
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelcolor': INK,
        'ytick.labelcolor': INK,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'lines.linewidth': 1.0,
        'lines.markersize': 3.5,
        'lines.markeredgecolor': 'white',    # surface ring around markers
        'lines.markeredgewidth': 0.5,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        'legend.frameon': False,
        'legend.handlelength': 2.6,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'figure.constrained_layout.use': True,
        'pdf.compression': 9,
    })


def figure(height_mm, width=COLUMN_WIDTH, **kwargs):
    """A figure of the given width (default: one IEEE column) and height in mm."""
    return plt.subplots(figsize=(width, height_mm * MM), **kwargs)


def save(fig, path, title):
    """Vector PDF with embedded fonts; the size is the figure size, so 100 % in LaTeX."""
    fig.savefig(path, format='pdf', metadata={'Title': title, 'Creator': 'interact evaluation'})
    plt.close(fig)
