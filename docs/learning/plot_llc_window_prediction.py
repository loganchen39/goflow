"""LLC prediction over the 512x768 window (rows 0-512, cols 233-1001), truth vs model.

The UNet is fully convolutional, so it runs on the whole window in one pass (512 and 768
are divisible by 16). Inputs are built exactly as SSTDataset does: 3 consecutive frames of
loggrad_T, normalised (v + 19) / 19; the target is U/V at the middle frame.
Figure: one sample (target frame 3001) with the training/test boxes outlined.
Table: R^2 per box over NS samples spread through the record, compared with the
256x256-tile test result. Run from the repo root:
    env -u PYTHONPATH /glade/work/lgchen/conda-envs/neurost/bin/python docs/learning/plot_llc_window_prediction.py
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from netCDF4 import Dataset
from dataSST import lgtMin, lgtMax
from goflow_core import initialize_model, dx_kernel, dy_kernel, compute_velocity_gradients, compute_derived_fields

LLC = 'llcGoes_gradT_trunc.nc'
CKPT = '/glade/derecho/scratch/lgchen/goflow_runs/train_20260929/lgt_unet16_1_3_{}cs.pth'
W = (0, 512, 233, 1001)                     # valid_inds
T_PLOT = 3000                               # first frame -> target frame 3001 (step-2 sample)
NS = 60
OUT = 'docs/figures/learning_step8_llc_window.png'
Nx = 1001
TRAIN = [(0, 256, 256, 512), (0, 256, 512, 768), (256, 512, 256, 512), (256, 512, 512, 768), (256, 512, Nx - 256, Nx)]
TEST = (0, 256, Nx - 256, Nx)
torch.set_num_threads(8)

nc = Dataset(LLC)
S = (slice(W[0], W[1]), slice(W[2], W[3]))
kx, ky = dx_kernel(5.0), dy_kernel(5.0)
models = {}
for cs in ('0.0', '0.2'):
    m = initialize_model(3, 2, 'unet', 16, device=torch.device('cpu'))
    m.load_state_dict(torch.load(CKPT.format(cs), map_location='cpu')); m.eval(); models[cs] = m


def sample(t0):
    x = np.stack([np.ma.filled(nc['loggrad_T'][(t0 + i,) + S], np.nan) for i in range(3)])
    x = np.nan_to_num(((x - lgtMin) / (lgtMax - lgtMin)).astype(np.float32))
    y = np.stack([np.ma.filled(nc[v][(t0 + 1,) + S], np.nan) for v in ('U', 'V')]).astype(np.float32)
    return x, y


def vort(uv):
    return compute_derived_fields(*compute_velocity_gradients(torch.from_numpy(uv)[None], kx, ky))[0][0, 0].numpy()


land = np.ma.filled(nc['loggrad_T'][(0,) + S], np.nan) == 0
local = lambda b: (slice(b[0] - W[0], b[1] - W[0]), slice(b[2] - W[2], b[3] - W[2]))
C = 8                                                       # interior crop, as in REPRODUCE_DERECHO.md §7

# --- R^2 per box over NS samples --------------------------------------------------------
boxes = {f'train {i + 1}': b for i, b in enumerate(TRAIN)}; boxes['TEST'] = TEST
acc = {cs: {k: {f: [0., 0., 0., 0] for f in ('uv', 'vort')} for k in boxes} for cs in models}


def add(a, t, p):
    ok = np.isfinite(t) & np.isfinite(p); t, p = t[ok].astype(float), p[ok].astype(float)
    a[0] += ((t - p) ** 2).sum(); a[1] += (t ** 2).sum(); a[2] += t.sum(); a[3] += t.size


starts = np.linspace(0, 8226, NS).astype(int) // 3 * 3          # non-overlapping sample starts
with torch.no_grad():
    for t0 in starts:
        x, y = sample(t0); vt = vort(y)
        for cs, m in models.items():
            p = m(torch.from_numpy(x)[None])[0].numpy(); vp = vort(p)
            for k, b in boxes.items():
                r, c = local(b); r = slice(r.start + C, r.stop - C); c = slice(c.start + C, c.stop - C)
                ocean = ~land[r, c]
                add(acc[cs][k]['uv'], y[:, r, c][:, ocean], p[:, r, c][:, ocean])
                add(acc[cs][k]['vort'], vt[r, c][ocean], vp[r, c][ocean])
r2 = lambda a: 1 - a[0] / (a[1] - a[2] ** 2 / a[3])
print(f'R^2 per box from 512x768-window inference, {NS} samples, ocean pixels, {C}-px interior crop')
print(f'{"box":8s} | {"stage0 U/V":>10s} {"stage0 vort":>11s} | {"stage1 U/V":>10s} {"stage1 vort":>11s}')
for k in boxes:
    print(f'{k:8s} | {r2(acc["0.0"][k]["uv"]):10.3f} {r2(acc["0.0"][k]["vort"]):11.3f} | '
          f'{r2(acc["0.2"][k]["uv"]):10.3f} {r2(acc["0.2"][k]["vort"]):11.3f}')
print('U/V error size vs signal size (stage 0): RMSE and truth std, m/s')
for k in boxes:
    a = acc['0.0'][k]['uv']
    print(f'{k:8s} | RMSE {np.sqrt(a[0] / a[3]):.3f} | truth std {np.sqrt(a[1] / a[3] - (a[2] / a[3]) ** 2):.3f}')

# --- figure: one sample ---------------------------------------------------------------------
x, y = sample(T_PLOT)
with torch.no_grad():
    p0 = models['0.0'](torch.from_numpy(x)[None])[0].numpy()
    p1 = models['0.2'](torch.from_numpy(x)[None])[0].numpy()
L = lambda a: np.where(land, np.nan, a)
sp_t, sp_0, sp_1 = (L(np.hypot(*a)) for a in (y, p0, p1))
v_t, v_0, v_1 = L(vort(y)), L(vort(p0)), L(vort(p1))
with Dataset(LLC) as d2:
    lat = d2['lat'][W[0]:W[1]]; lon = d2['lon'][W[2]:W[3]]
ext = [lon[0], lon[-1], lat[0], lat[-1]]

INK, MUTED, BLUE, ORANGE = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK})
fig, axs = plt.subplots(3, 3, figsize=(18, 12.6), constrained_layout=True)


def outline(ax):
    for b, col, lw in [(b, BLUE, 1.3) for b in TRAIN] + [(TEST, ORANGE, 2.2)]:
        ax.add_patch(Rectangle((lon[0] + 0.02 * (b[2] - W[2]), lat[0] + 0.02 * (b[0] - W[0])),
                               0.02 * (b[3] - b[2]), 0.02 * (b[1] - b[0]), fill=False, ec=col, lw=lw))
    ax.set_facecolor('#d8d7d2'); ax.set_xlabel('lon (°E)'); ax.set_ylabel('lat (°N)')


smax = np.nanpercentile(sp_t, 99.5); vmax = np.nanpercentile(np.abs(v_t), 99)
emax = np.nanpercentile(np.abs(sp_0 - sp_t), 99)
rows = [('Speed (m/s)', sp_t, sp_0, sp_1, 'Blues', 0, smax),
        ('Vorticity (model units)', v_t, v_0, v_1, 'RdBu_r', -vmax, vmax)]
for i, (name, t, a, b, cmap, lo, hi) in enumerate(rows):
    for j, (fld, ttl) in enumerate([(t, 'Truth (LLC)'), (a, 'Stage 0 prediction'), (b, 'Stage 1 prediction')]):
        ax = axs[i, j]; im = ax.imshow(fld, origin='lower', extent=ext, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(f'{name}: {ttl}'); outline(ax)
    fig.colorbar(im, ax=axs[i, 2], shrink=0.85)
for j, (fld, ttl) in enumerate([(np.hypot(*(p0 - y)), 'Stage 0 |velocity error|'),
                                (np.hypot(*(p1 - y)), 'Stage 1 |velocity error|')]):
    ax = axs[2, j]; im = ax.imshow(L(fld), origin='lower', extent=ext, cmap='Oranges', vmin=0, vmax=emax)
    ax.set_title(f'{ttl} (m/s)'); outline(ax)
fig.colorbar(im, ax=axs[2, 1], shrink=0.85)
ax = axs[2, 2]
im = ax.imshow(x[1], origin='lower', extent=ext, cmap='Greys',
               vmin=np.percentile(x[1][~land], 1), vmax=np.percentile(x[1][~land], 99.5))
ax.set_title('Input x[1]: normalised loggrad_T (target frame)'); outline(ax)
fig.suptitle(f'LLC, whole 512×768 window in one pass (rows 0–512, cols 233–1001), target frame {T_PLOT + 1}.  '
             'Blue: training boxes; orange: test box', color=INK, fontsize=12, fontweight='bold')
fig.savefig(OUT, dpi=100)
print('saved', OUT)
