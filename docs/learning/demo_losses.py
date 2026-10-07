"""What do the two training losses reward?

Takes a batch of 32 consecutive training samples (the step-2 box), and evaluates the
exact training losses -- masked L1 (train_goflow.py:138) and spectral_loss
(spectral_loss.py:107) -- for hand-made "predictions" (truth smoothed, shifted,
phase-scrambled, all zero) and for the trained stage 0 / stage 1 models.
Also splits the spectral loss by wavelength and compares the gradient sizes of the
two loss terms. Run from the repo root:
    env -u PYTHONPATH /glade/work/lgchen/conda-envs/neurost/bin/python docs/learning/demo_losses.py
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dataSST import SSTDataset
from goflow_core import initialize_model, create_boundary_mask, create_tukey_window
from spectral_loss import spectral_loss, compute_ke_spectrum_2d

torch.manual_seed(0); rng = np.random.default_rng(0)
LLC = 'llcGoes_gradT_trunc.nc'
BOX, I0, NB = (0, 256, 512, 768), 1000, 32
CKPT = '/glade/derecho/scratch/lgchen/goflow_runs/train_20260929/lgt_unet16_1_3_{}cs.pth'
OUT = 'docs/figures/learning_step4_losses.png'

ds = SSTDataset(LLC, ['loggrad_T', 'U', 'V'], BOX, step0=1, num_input_frames=3)
xs, ys = zip(*[ds[i] for i in range(I0, I0 + NB)])
x = torch.from_numpy(np.stack(xs)); y = torch.from_numpy(np.stack(ys))      # (32,3,256,256), (32,2,256,256)
mask = create_boundary_mask((256, 256)); tukey = create_tukey_window((256, 256))
crit = nn.L1Loss()


def l1(pred):                       # exactly train_goflow.py:138
    return crit(y.squeeze() * mask[None, None], pred.squeeze() * mask[None, None]).item()


def spec(pred):
    return spectral_loss(pred, y, tukey).item()


def phase_scramble(a):
    F = np.fft.rfft2(a, axes=(-2, -1))
    ph = np.exp(2j * np.pi * rng.random(F.shape))
    return np.fft.irfft2(np.abs(F) * ph, s=a.shape[-2:], axes=(-2, -1)).astype(np.float32)


yn = y.numpy()
models = {}
for cs in ('0.0', '0.2'):
    m = initialize_model(3, 2, 'unet', 16, device=torch.device('cpu'))
    m.load_state_dict(torch.load(CKPT.format(cs), map_location='cpu')); m.eval()
    models[cs] = m
with torch.no_grad():
    p0 = models['0.0'](x); p1 = models['0.2'](x)

cases = {
    'truth': y,
    'truth blurred σ=2 px': torch.from_numpy(gaussian_filter(yn, (0, 0, 2, 2))),
    'truth blurred σ=4 px': torch.from_numpy(gaussian_filter(yn, (0, 0, 4, 4))),
    'truth shifted 4 px': torch.roll(y, (4, 4), dims=(2, 3)),
    'truth phase-scrambled': torch.from_numpy(phase_scramble(yn)),
    'all zeros': torch.zeros_like(y),
    'stage 0 model': p0,
    'stage 1 model': p1,
}
res = {k: (l1(v), spec(v)) for k, v in cases.items()}
print(f'{"prediction":24s} {"L1 (m/s)":>9s} {"spectral":>9s}')
for k, (a, b) in res.items():
    print(f'{k:24s} {a:9.4f} {b:9.3f}')

# --- spectral loss split by wavelength of each rfft2 bin --------------------------------
ky = np.fft.fftfreq(256)[:, None]; kx = np.fft.rfftfreq(256)[None, :]
kr = np.hypot(kx, ky); wl = np.where(kr > 0, 1 / np.maximum(kr, 1e-12), np.inf)
BANDS = [('>64', 64, np.inf), ('32–64', 32, 64), ('16–32', 16, 32), ('8–16', 8, 16), ('4–8', 4, 8), ('<4', 0, 4)]
band_of = [((wl >= lo) & (wl < hi)) for _, lo, hi in BANDS]
nbin = np.array([b.sum() for b in band_of]); nbin_share = nbin / nbin.sum()


def per_bin_loss(pred):
    u, v = pred[:, 0] * tukey, pred[:, 1] * tukey
    ut, vt = y[:, 0] * tukey, y[:, 1] * tukey
    d = torch.log(compute_ke_spectrum_2d(u, v) + 1e-10) - torch.log(compute_ke_spectrum_2d(ut, vt) + 1e-10)
    return (d ** 2).mean(0).numpy()                                  # (256, 129)


share = {}
for k in ('stage 0 model', 'stage 1 model'):
    pb = per_bin_loss(cases[k]); share[k] = np.array([pb[b].sum() for b in band_of]) / pb.sum()
    print(f'{k}: share of spectral loss by wavelength band (px):',
          ' '.join(f'{n}:{s:.2f}' for (n, _, _), s in zip(BANDS, share[k])))
print('share of rfft2 bins by band:', ' '.join(f'{n}:{s:.2f}' for (n, _, _), s in zip(BANDS, nbin_share)))
pb = per_bin_loss(cases['truth phase-scrambled'])
print('phase-scrambled truth, share by band:',
      ' '.join(f'{n}:{s:.2f}' for (n, _, _), s in zip(BANDS, np.array([pb[b].sum() for b in band_of]) / pb.sum())))

# --- which term drives the stage 1 update? ------------------------------------------------
# A realistic training batch: 64 random samples from all 5 training boxes (shuffled, as the
# DataLoader does). Using the 32 consecutive samples above in train mode would give BatchNorm
# unrepresentative batch statistics.
TRAIN_BOXES = [(0, 256, 256, 512), (0, 256, 512, 768), (256, 512, 256, 512),
               (256, 512, 512, 768), (256, 512, 745, 1001)]
dsets = [SSTDataset(LLC, ['loggrad_T', 'U', 'V'], b, step0=1, num_input_frames=3) for b in TRAIN_BOXES]
pick = [(rng.integers(5), rng.integers(2742)) for _ in range(64)]
xb_, yb_ = zip(*[dsets[b][i] for b, i in pick])
xb = torch.from_numpy(np.stack(xb_)); yb = torch.from_numpy(np.stack(yb_))
m = models['0.2']; m.train()                # train mode, as in train_epoch
pred = m(xb)
L1t = crit(yb.squeeze() * mask[None, None], pred.squeeze() * mask[None, None])
SPt = spectral_loss(pred, yb, tukey)
params = [p for p in m.parameters() if p.requires_grad]
g1 = torch.autograd.grad(0.8 * L1t, params, retain_graph=True)
g2 = torch.autograd.grad(0.2 * SPt, params)
n1 = torch.sqrt(sum((g ** 2).sum() for g in g1)).item(); n2 = torch.sqrt(sum((g ** 2).sum() for g in g2)).item()
cos = (sum((a * b).sum() for a, b in zip(g1, g2)) / (n1 * n2)).item()
print(f'stage 1 weights, c_spec=0.2, random batch of 64: value 0.8*L1={0.8 * L1t.item():.4f}  0.2*spec={0.2 * SPt.item():.4f} | '
      f'grad norm 0.8*L1={n1:.4f}  0.2*spec={n2:.4f}  ratio spec/L1={n2 / n1:.2f}  cosine={cos:.3f}')

# --- figure -------------------------------------------------------------------------------
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK, 'axes.spines.top': False,
                     'axes.spines.right': False})
col = {k: '#9a9890' for k in cases}; col['stage 0 model'] = '#2a78d6'; col['stage 1 model'] = '#eb6834'
names = list(cases)[1:]                     # truth is (0, 0); leave it out of the bars
fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True,
                        gridspec_kw={'width_ratios': [1, 1, 1.15]})
for j, (ttl, idx, fmt) in enumerate((('Masked L1 loss (m/s): rewards being in the right place', 0, '{:.3f}'),
                                     ('Spectral loss: rewards the right amount of energy per scale', 1, '{:.2f}'))):
    ax = axs[j]; vals = [res[k][idx] for k in names]
    ax.barh(names, vals, color=[col[k] for k in names], height=0.62)
    for i, v in enumerate(vals):
        ax.text(v, i, ' ' + fmt.format(v), va='center', color=INK, fontsize=8.5)
    ax.invert_yaxis(); ax.set_title(ttl); ax.grid(True, axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(0, max(vals) * 1.18)
    if j == 1:
        ax.set_yticklabels([]); ax.set_xscale('log'); ax.set_xlim(0.01, max(vals) * 6)
        ax.set_xlabel('log scale')
ax = axs[2]; w = 0.27; xb = np.arange(len(BANDS))
ax.bar(xb - w, nbin_share, w, color='#c3c2b7', label='share of FFT bins')
ax.bar(xb, share['stage 0 model'], w, color='#2a78d6', label='share of stage 0 spectral loss')
ax.bar(xb + w, share['stage 1 model'], w, color='#eb6834', label='share of stage 1 spectral loss')
ax.set_xticks(xb, [n for n, _, _ in BANDS]); ax.set_xlabel('wavelength band (px; 1 px ≈ 1.95 km)')
ax.set_ylabel('share'); ax.legend(frameon=False, loc='upper left'); ax.grid(True, axis='y', color=GRID, lw=0.6)
ax.set_axisbelow(True); ax.set_title('Where the spectral loss comes from')
fig.suptitle(f'Training losses on {NB} training samples (box rows {BOX[0]}–{BOX[1]}, cols {BOX[2]}–{BOX[3]}, '
             f'samples {I0}–{I0 + NB - 1}); a perfect prediction scores 0 on both',
             color=INK, fontsize=11, fontweight='bold')
fig.savefig(OUT, dpi=110)
print('saved', OUT)
