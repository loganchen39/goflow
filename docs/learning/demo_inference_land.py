"""Does the land encoding at GOES inference matter?

In LLC training data land has loggrad_T = 0 -> normalised input 1.0.
In the GOES file land is NaN -> SatelliteDataset's nan_to_num makes it 0.0.
This runs the stage 0 model on GOES samples both ways (land = 0 as inference does,
land = 1 as in training) and measures how much the predicted ocean velocity changes
with distance from the coast. Also plots one frame of the inference inputs/outputs.
Run from the repo root:
    env -u PYTHONPATH /glade/work/lgchen/conda-envs/neurost/bin/python docs/learning/demo_inference_land.py
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
from dataSST import SatelliteDataset
from goflow_core import initialize_model

G = 'GS_BT_NESMA2023_HiRes_SUBSECTION_grad_mask.nc'
VI = (0, 512, 233, 1001)                    # valid_inds used by inf_llc_stage1.py
CKPT = '/glade/derecho/scratch/lgchen/goflow_runs/train_20260929/lgt_unet16_1_3_0.0cs.pth'
SAMPLES = list(range(0, 575, 48))           # 12 samples spread over the 2 days
OUT = 'docs/figures/learning_step5_inference.png'

m = initialize_model(3, 2, 'unet', 16, device=torch.device('cpu'))
m.load_state_dict(torch.load(CKPT, map_location='cpu')); m.eval()
ds = SatelliteDataset(G, ['log_gradT'], VI, train=False)
g = Dataset(G)
S = (slice(VI[0], VI[1]), slice(VI[2], VI[3]))
land = np.isnan(np.ma.filled(g['log_gradT'][(0,) + S], np.nan))       # constant in time
dist = distance_transform_edt(~land)                                   # px from nearest land

edges = [0, 4, 8, 16, 32, 64, 1e9]
diffs = {i: [] for i in range(len(edges) - 1)}
speed_all = []
with torch.no_grad():
    for k in SAMPLES:
        x = torch.from_numpy(ds[k])[None]                              # land = 0 (as inference)
        x1 = x.clone(); x1[:, :, torch.from_numpy(land)] = 1.0         # land = 1 (as training)
        p0, p1 = m(x)[0].numpy(), m(x1)[0].numpy()
        d = np.hypot(*(p0 - p1)); sp = np.hypot(*p1)
        speed_all.append(sp[~land].mean())
        for i in range(len(edges) - 1):
            sel = (~land) & (dist > edges[i]) & (dist <= edges[i + 1])
            diffs[i].append(d[sel].mean())
        if k == 288:
            keep = dict(x=x[0, 1].numpy(), p0=p0, p1=p1, d=d, k=k)
print(f'mean ocean speed (land=1 run): {np.mean(speed_all):.3f} m/s')
print('mean |Δ velocity| (land=0 vs land=1) by distance from land:')
for i in range(len(edges) - 1):
    hi = '∞' if edges[i + 1] > 1e8 else edges[i + 1]
    print(f'  {edges[i]}–{hi} px: {np.mean(diffs[i]):.3f} m/s')

# --- one-frame figure ---------------------------------------------------------------------
k = keep['k']; f_mid = k + 13
mask = np.ma.filled(g['mask'][(f_mid,) + S], np.nan)
lat = g['lat'][VI[0]:VI[1]]; lon = g['lon'][VI[2]:VI[3]]
ext = [lon[0], lon[-1], lat[0], lat[-1]]
INK, MUTED = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK})
fig, axs = plt.subplots(2, 2, figsize=(14, 9.4), constrained_layout=True)
ax = axs[0, 0]
v = keep['x']; vv = v[v > 0]
im = ax.imshow(v, origin='lower', extent=ext, cmap='Greys', vmin=np.percentile(vv, 1), vmax=np.percentile(vv, 99.5))
ax.set_title(f'Model input x[1]: GOES log_gradT, frame {f_mid} (normalised)\n'
             'land = 0.0 here (NaN → 0); clouds are NOT removed')
fig.colorbar(im, ax=ax, shrink=0.85)
ax = axs[0, 1]
im = ax.imshow(np.where(land, np.nan, mask), origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=1.4)
ax.set_title(f'Clear-sky mask, frame {f_mid}: {100 * np.nanmean(np.where(land, np.nan, mask)):.0f}% of ocean clear\n'
             '(applied only to the OUTPUT; masked pixels are written as 0)')
ax = axs[1, 0]
sp = np.hypot(*keep['p0'])
im = ax.imshow(np.where(land, np.nan, sp), origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=np.percentile(sp[~land], 99.5))
ax.contour(lon, lat, mask, levels=[0.5], colors=[INK], linewidths=0.5)
ax.set_title('Predicted speed (stage 0), before masking; black line = cloud edge')
fig.colorbar(im, ax=ax, shrink=0.85, label='m/s')
ax = axs[1, 1]
im = ax.imshow(np.where(land, np.nan, keep['d']), origin='lower', extent=ext, cmap='Oranges', vmin=0,
               vmax=np.percentile(keep['d'][~land], 99))
ax.set_title('|Δ velocity| if land were encoded as in training (1.0 instead of 0.0)')
fig.colorbar(im, ax=ax, shrink=0.85, label='m/s')
for a in axs.flat:
    a.set_xlabel('lon (°E)'); a.set_ylabel('lat (°N)'); a.set_facecolor('#d8d7d2')
fig.suptitle(f'GOES inference, sample {k} (frames {k + 1}, {k + 13}, {k + 25}), window rows 0–512, cols 233–1001',
             color=INK, fontsize=12, fontweight='bold')
fig.savefig(OUT, dpi=105)
print('saved', OUT)
