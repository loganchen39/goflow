"""Plot one GOFLOW training sample exactly as train_goflow.py sees it.

Uses SSTDataset with the training settings (step0=1, nframes=3, non-overlapping),
so x is the normalised 3-frame loggrad_T input and y is the U/V target at the
middle frame. Run from the repo root:
    env -u PYTHONPATH /glade/work/lgchen/conda-envs/neurost/bin/python docs/learning/plot_training_sample.py
"""
import os, sys
sys.path.insert(0, os.getcwd())   # repo root, for dataSST / goflow_core
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from dataSST import SSTDataset, lgtMin, lgtMax
from goflow_core import dx_kernel, dy_kernel, compute_velocity_gradients, compute_derived_fields

LLC = 'llcGoes_gradT_trunc.nc'
BOX = (0, 256, 512, 768)          # training box 2 of 5 (Gulf Stream)
LAND_BOX = (256, 512, 256, 512)   # training box 3 of 5 (mostly land/fill)
IDX = 1000                        # sample index -> frames 3000, 3001, 3002
OUT = 'docs/figures/learning_step2_training_sample.png'

ds = SSTDataset(LLC, ['loggrad_T', 'U', 'V'], BOX, step0=1, num_input_frames=3)
x, y = ds[IDX]                    # x: (3,256,256) normalised; y: (2,256,256) m/s
t0 = IDX * 3                      # non-overlapping indexing in SSTDataset

# vorticity of the target, computed the same way as the training metric (pm=pn=5)
yt = torch.from_numpy(y)[None]
vort = compute_derived_fields(*compute_velocity_gradients(yt, dx_kernel(5.0), dy_kernel(5.0)))[0][0, 0].numpy()

with Dataset(LLC) as nc:
    lat = nc['lat'][BOX[0]:BOX[1]]; lon = nc['lon'][BOX[2]:BOX[3]]
    llat = nc['lat'][LAND_BOX[0]:LAND_BOX[1]]; llon = nc['lon'][LAND_BOX[2]:LAND_BOX[3]]
ext = [lon[0], lon[-1], lat[0], lat[-1]]
lext = [llon[0], llon[-1], llat[0], llat[-1]]
xl, _ = SSTDataset(LLC, ['loggrad_T', 'U', 'V'], LAND_BOX, step0=1, num_input_frames=3)[IDX]

INK, MUTED = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK})
fig, axs = plt.subplots(2, 4, figsize=(17, 8.6), constrained_layout=True)

vmin, vmax = np.percentile(x[x < 1], [1, 99.5])
for k in range(3):
    ax = axs[0, k]
    im = ax.imshow(x[k], origin='lower', extent=ext, cmap='Greys', vmin=vmin, vmax=vmax)
    tag = ' (target time)' if k == 1 else ''
    ax.set_title(f'Input x[{k}]: loggrad_T, frame t={t0 + k}{tag}')
fig.colorbar(im, ax=axs[0, 2], shrink=0.85, label='(loggrad_T + 19) / 19   (same scale for x[0..2])')

ax = axs[0, 3]
d = x[2] - x[0]; s = np.percentile(np.abs(d), 99)
im = ax.imshow(d, origin='lower', extent=ext, cmap='RdBu_r', vmin=-s, vmax=s)
ax.set_title('x[2] − x[0]: change over 2 frames\n(the motion signal the model can use)')
fig.colorbar(im, ax=ax, shrink=0.85)

s = np.percentile(np.abs(y), 99.5)
for k, name in enumerate(['U (eastward)', 'V (northward)']):
    ax = axs[1, k]
    im = ax.imshow(y[k], origin='lower', extent=ext, cmap='RdBu_r', vmin=-s, vmax=s)
    ax.set_title(f'Target y[{k}]: {name}, frame t={t0 + 1}')
fig.colorbar(im, ax=axs[1, 1], shrink=0.85, label='m/s   (same scale for U and V)')

ax = axs[1, 2]
sp = np.hypot(y[0], y[1])
im = ax.imshow(sp, origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=np.percentile(sp, 99.5))
q = 12
LO, LA = np.meshgrid(lon, lat)
ax.quiver(LO[::q, ::q], LA[::q, ::q], y[0, ::q, ::q], y[1, ::q, ::q], color=INK, scale=25, width=0.003)
ax.set_title('Target speed + velocity arrows')
fig.colorbar(im, ax=ax, shrink=0.85, label='m/s')

ax = axs[1, 3]
s = np.percentile(np.abs(vort), 99)
im = ax.imshow(vort, origin='lower', extent=ext, cmap='RdBu_r', vmin=-s, vmax=s)
ax.set_title('Target vorticity (as in the gradR² metric)')
fig.colorbar(im, ax=ax, shrink=0.85, label='model units (pm = pn = 5)')

for ax in axs.flat:
    ax.set_xlabel('lon (°E)'); ax.set_ylabel('lat (°N)')
fig.suptitle(f'One training sample: box rows {BOX[0]}–{BOX[1]}, cols {BOX[2]}–{BOX[3]}, '
             f'sample {IDX} (frames {t0}–{t0 + 2}).  x shape {x.shape}, y shape {y.shape}',
             color=INK, fontsize=12, fontweight='bold')
fig.savefig(OUT, dpi=110)

# small companion figure: the land-dominated training box
fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
im = ax.imshow(xl[1], origin='lower', extent=lext, cmap='Greys', vmin=vmin, vmax=1)
ax.set_title(f'Training box rows {LAND_BOX[0]}–{LAND_BOX[1]}, cols {LAND_BOX[2]}–{LAND_BOX[3]}\n'
             f'input x[1]: {100 * (xl[1] == 1).mean():.0f}% of pixels = 1.0 (land fill)')
ax.set_xlabel('lon (°E)'); ax.set_ylabel('lat (°N)')
fig.colorbar(im, ax=ax, shrink=0.85)
fig.savefig(OUT.replace('.png', '_landbox.png'), dpi=110)

print('frames', t0, t0 + 1, t0 + 2, '| x range', x.min(), x.max(), '| U,V range', y.min(), y.max(),
      '| speed mean/max', sp.mean(), sp.max())
yl = SSTDataset(LLC, ['loggrad_T', 'U', 'V'], LAND_BOX, step0=1, num_input_frames=3)[IDX][1]
print('land box: frac x==1', (xl[1] == 1).mean(), '| |U|,|V| max where x==1',
      np.abs(yl[:, xl[1] == 1]).max())
