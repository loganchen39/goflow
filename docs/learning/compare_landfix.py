"""Compare GOES predictions before and after the land-encoding fix.

old: infer_retrained_20260929 (land fed to the model as 0.0)
new: infer_landfix_20261007   (land fed as 1.0, as in LLC training)
Only clear-sky ocean pixels are compared (the output files are multiplied by the cloud
mask, so cloudy pixels are 0 in both). Records 575-596 are empty and skipped.
Run from the repo root:
    env -u PYTHONPATH /glade/work/lgchen/conda-envs/neurost/bin/python docs/learning/compare_landfix.py
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt

R = '/glade/derecho/scratch/lgchen/goflow_runs/'
G = 'GS_BT_NESMA2023_HiRes_SUBSECTION_grad_mask.nc'
F = 'preds_lgt_unet16_1_3_{cs}cs_' + G
RUNS = {'old': R + 'infer_retrained_20260929/stage_{cs}cs/', 'new': R + 'infer_landfix_20261007/stage_{cs}cs/'}
NT, CH = 575, 50
EDGES = [0, 4, 8, 16, 32, 64, 1e9]
LABELS = ['0–4', '4–8', '8–16', '16–32', '32–64', '>64']
OUT = 'docs/figures/learning_step5_landfix.png'

with Dataset(G) as g:
    land = np.isnan(np.ma.filled(g['log_gradT'][0, 0:512, 233:1001], np.nan))
    lat = g['lat'][0:512]; lon = g['lon'][233:1001]
dist = distance_transform_edt(~land)
band = np.digitize(dist, EDGES[1:-1]); band[land] = -1


def stats(cs):
    o = Dataset(RUNS['old'].format(cs=cs) + F.format(cs=cs)); n = Dataset(RUNS['new'].format(cs=cs) + F.format(cs=cs))
    nb = len(LABELS)
    s_old = np.zeros(nb); s_new = np.zeros(nb); dsum = np.zeros(nb); cnt = np.zeros(nb)
    dmap = np.zeros(land.shape); spo = np.zeros(land.shape); spn = np.zeros(land.shape); cmap_ = np.zeros(land.shape)
    for i in range(0, NT, CH):
        sl = slice(i, min(NT, i + CH))
        # cloud-masked pixels are 0 in the old files and NaN in files written after 2026-10-07
        uo, vo = (np.nan_to_num(np.ma.filled(o[v][sl], 0.)) for v in ('U', 'V'))
        un, vn = (np.nan_to_num(np.ma.filled(n[v][sl], 0.)) for v in ('U', 'V'))
        clear = (uo != 0) | (vo != 0) | (un != 0) | (vn != 0)
        clear &= ~land[None]
        so, sn = np.hypot(uo, vo), np.hypot(un, vn); d = np.hypot(uo - un, vo - vn)
        for b in range(nb):
            sel = clear & (band[None] == b)
            s_old[b] += so[sel].sum(); s_new[b] += sn[sel].sum(); dsum[b] += d[sel].sum(); cnt[b] += sel.sum()
        dmap += (d * clear).sum(0); spo += (so * clear).sum(0); spn += (sn * clear).sum(0); cmap_ += clear.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return dict(s_old=s_old / cnt, s_new=s_new / cnt, d=dsum / cnt, frac=cnt / cnt.sum(),
                    dmap=dmap / cmap_, spo=spo / cmap_, spn=spn / cmap_, nclear=cmap_)


res = {cs: stats(cs) for cs in ('0.0', '0.2')}
for cs, r in res.items():
    print(f'stage {cs}cs (clear-sky ocean pixels, all 575 records)')
    print(f'  {"px from land":>12s} {"share":>6s} {"speed old":>9s} {"speed new":>9s} {"|Δvel|":>7s}')
    for b, lab in enumerate(LABELS):
        print(f'  {lab:>12s} {r["frac"][b]:6.2f} {r["s_old"][b]:9.3f} {r["s_new"][b]:9.3f} {r["d"][b]:7.3f}')
    allw = r['frac']
    print(f'  {"all":>12s} {1:6.2f} {np.sum(r["s_old"] * allw):9.3f} {np.sum(r["s_new"] * allw):9.3f} {np.sum(r["d"] * allw):7.3f}')

# --- figure (stage 0) ---------------------------------------------------------------------
r = res['0.0']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK})
fig = plt.figure(figsize=(16, 9.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ext = [lon[0], lon[-1], lat[0], lat[-1]]
few = r['nclear'] < 20                                   # hide pixels clear in <20 records
vmax = np.nanpercentile(np.where(few | land, np.nan, r['spo']), 99.5)
for j, (key, ttl) in enumerate((('spo', 'Old: land fed as 0.0'), ('spn', 'New: land fed as 1.0 (as in training)'))):
    ax = fig.add_subplot(gs[0, j])
    im = ax.imshow(np.where(few | land, np.nan, r[key]), origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=vmax)
    ax.set_title(f'{ttl}\nmean predicted speed over clear-sky records (stage 0)')
    ax.set_facecolor('#d8d7d2'); ax.set_xlabel('lon (°E)'); ax.set_ylabel('lat (°N)')
fig.colorbar(im, ax=ax, shrink=0.85, label='m/s')
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(np.where(few | land, np.nan, r['dmap']), origin='lower', extent=ext, cmap='Oranges', vmin=0,
               vmax=np.nanpercentile(np.where(few | land, np.nan, r['dmap']), 99.5))
ax.set_title('Mean |Δ velocity| new vs old\n(clear-sky records)')
ax.set_facecolor('#d8d7d2'); ax.set_xlabel('lon (°E)'); ax.set_ylabel('lat (°N)')
fig.colorbar(im, ax=ax, shrink=0.85, label='m/s')

ax = fig.add_subplot(gs[1, 0:2])
xb = np.arange(len(LABELS)); w = 0.2
for k, (cs, lab0) in enumerate((('0.0', 'stage 0'), ('0.2', 'stage 1'))):
    rr = res[cs]
    ax.bar(xb + (2 * k - 1.5) * w, rr['s_old'], w, color=['#9ec5f4', '#f6b89a'][k], label=f'{lab0}, old (land 0.0)')
    ax.bar(xb + (2 * k - 0.5) * w, rr['s_new'], w, color=['#2a78d6', '#eb6834'][k], label=f'{lab0}, new (land 1.0)')
ax.set_xticks(xb, LABELS); ax.set_xlabel('distance from land (px; 1 px ≈ 1.95 km)'); ax.set_ylabel('mean speed (m/s)')
ax.set_title('Mean predicted speed on clear-sky ocean pixels, by distance from the coast')
ax.legend(frameon=False, ncol=2, loc='upper right'); ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax = fig.add_subplot(gs[1, 2])
ax.bar(xb, res['0.0']['frac'], 0.6, color='#9a9890')
ax.set_xticks(xb, LABELS); ax.set_xlabel('distance from land (px)'); ax.set_ylabel('share of clear-sky ocean pixels')
ax.set_title('Where the clear-sky data are'); ax.grid(True, axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('GOES inference before and after the land-encoding fix (575 records, 2023-05-14 to 05-16)',
             color=INK, fontsize=12, fontweight='bold')
fig.savefig(OUT, dpi=105)
print('saved', OUT)
