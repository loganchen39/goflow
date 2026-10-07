"""Training / test boxes and the GOES inference window over the whole LLC domain.

Background: loggrad_T at one frame, land (loggrad_T == 0) in grey. Boxes are the
hard-coded ones in train_goflow.py:596-604 (Nx = 1001). Run from the repo root:
    env -u PYTHONPATH /glade/work/lgchen/conda-envs/neurost/bin/python docs/learning/plot_domain_boxes.py
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from netCDF4 import Dataset

LLC = 'llcGoes_gradT_trunc.nc'
FRAME = 3001
OUT = 'docs/figures/learning_step6_domain_boxes.png'

with Dataset(LLC) as nc:
    g = np.ma.filled(nc['loggrad_T'][FRAME], np.nan).astype(float)
    lat = nc['lat'][:]; lon = nc['lon'][:]
ny, nx = g.shape
Nx = nx
TRAIN = [(0, 256, 256, 512), (0, 256, 512, 768), (256, 512, 256, 512),
         (256, 512, 512, 768), (256, 512, Nx - 256, Nx)]
TEST = (0, 256, Nx - 256, Nx)
SAT = (0, 512, Nx - 768, Nx)                 # valid_inds: GOES inference window

land = g == 0
ocean = np.where(land, np.nan, g)

INK, MUTED = '#0b0b0b', '#52514e'
BLUE, ORANGE, LAND = '#2a78d6', '#eb6834', '#b9b8b2'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK})
fig, ax = plt.subplots(figsize=(15, 8.9), constrained_layout=True)
ax.imshow(np.where(land, 1, np.nan), origin='lower', cmap=matplotlib.colors.ListedColormap([LAND]),
          extent=[-0.5, nx - 0.5, -0.5, ny - 0.5])
v = ocean[np.isfinite(ocean)]
im = ax.imshow(ocean, origin='lower', cmap='Greys', vmin=np.percentile(v, 1), vmax=np.percentile(v, 99.5),
               extent=[-0.5, nx - 0.5, -0.5, ny - 0.5])
fig.colorbar(im, ax=ax, shrink=0.75, pad=0.01, label=f'loggrad_T (frame {FRAME}); land in grey')


def box(b, color, lw, ls='-', label=None, fill=False):
    r0, r1, c0, c1 = b
    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=fill, ec=color, fc=color if fill else 'none',
                           lw=lw, ls=ls, label=label, alpha=1 if not fill else 0.18, zorder=3))


box(SAT, INK, 1.6, '--', 'GOES inference window (valid_inds) 512×768')
for i, b in enumerate(TRAIN):
    box(b, BLUE, 2.6, label='training box (×5)' if i == 0 else None)
    r0, r1, c0, c1 = b
    landfrac = land[r0:r1, c0:c1].mean()
    ax.text(c0 + 8, r1 - 10, f'train {i + 1}\nrows {r0}–{r1}, cols {c0}–{c1}\nland {100 * landfrac:.0f}%',
            color=BLUE, fontsize=9, fontweight='bold', va='top', zorder=4,
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
box(TEST, ORANGE, 2.6, label='test box')
r0, r1, c0, c1 = TEST
ax.text(c1 - 8, r0 + 10, f'TEST\nrows {r0}–{r1}, cols {c0}–{c1}\nland {100 * land[r0:r1, c0:c1].mean():.0f}%',
        color=ORANGE, fontsize=9, fontweight='bold', ha='right', va='bottom', zorder=4,
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
# overlap of test with training box 2 (cols 745-768), and of train 4 with train 5
box((0, 256, Nx - 256, 768), ORANGE, 0, fill=True, label='overlap (23 columns)')
box((256, 512, Nx - 256, 768), BLUE, 0, fill=True)
# unused areas
ax.text(128, 256, 'cols 0–256\nnever used', ha='center', va='center', color=MUTED, fontsize=9, style='italic')
ax.text(500, 531, 'rows 512–551: never used', ha='center', va='center', color=MUTED, fontsize=9, style='italic',
        bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))

ax.set_xlim(-0.5, nx - 0.5); ax.set_ylim(-0.5, ny - 0.5)
ax.set_xlabel('column index (lon)'); ax.set_ylabel('row index (lat)')
ax.set_xticks(np.arange(0, nx, 128)); ax.set_yticks(np.arange(0, ny, 128))
sec = ax.secondary_xaxis('top', functions=(lambda c: lon[0] + 0.02 * c, lambda x: (x - lon[0]) / 0.02))
sec.set_xlabel('lon (°E)', color=MUTED)
secy = ax.secondary_yaxis('right', functions=(lambda r: lat[0] + 0.02 * r, lambda y: (y - lat[0]) / 0.02))
secy.set_ylabel('lat (°N)', color=MUTED)
ax.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
ax.set_title(f'LLC domain {ny}×{nx} (0.02°, ≈1.95 km): 5 training boxes, 1 test box, GOES window   '
             f'(loggrad_T frame {FRAME})', fontweight='bold')
fig.savefig(OUT, dpi=110)
print('land % by box:', [round(100 * land[a:b, c:d].mean(), 1) for a, b, c, d in TRAIN],
      'test', round(100 * land[TEST[0]:TEST[1], TEST[2]:TEST[3]].mean(), 1))
cov = np.zeros(land.shape, bool)
for a, b, c, d in TRAIN + [TEST]:
    cov[a:b, c:d] = True
print(f'ocean share of domain covered by any box: {cov[~land].mean():.2f}; '
      f'ocean in cols 0-256: {(~land[:, :256]).sum() / (~land).sum():.2f}; rows 512-551: {(~land[512:]).sum() / (~land).sum():.2f}')
print('saved', OUT)
