"""How far, and on which frames, does one UNet output pixel look?

Loads the trained stage 0 / stage 1 checkpoints, feeds the step-2 training sample,
and backpropagates from the U and V outputs at the centre pixel to the input.
|d output / d input| is the model's local sensitivity map (its effective receptive field).
Run from the repo root:
    env -u PYTHONPATH /glade/work/lgchen/conda-envs/neurost/bin/python docs/learning/plot_unet_sensitivity.py
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from dataSST import SSTDataset
from goflow_core import initialize_model

LLC = 'llcGoes_gradT_trunc.nc'
BOX, IDX = (0, 256, 512, 768), 1000          # same sample as step 2
CKPT = '/glade/derecho/scratch/lgchen/goflow_runs/train_20260929/lgt_unet16_1_3_{}cs.pth'
C = 128                                      # output pixel (centre of the tile)
OUT = 'docs/figures/learning_step3_unet_sensitivity.png'

x, _ = SSTDataset(LLC, ['loggrad_T', 'U', 'V'], BOX, step0=1, num_input_frames=3)[IDX]
r = np.hypot(*(np.indices((256, 256)) - C))


def sensitivity(cs):
    m = initialize_model(3, 2, 'unet', 16, device=torch.device('cpu'))
    m.load_state_dict(torch.load(CKPT.format(cs), map_location='cpu')); m.eval()
    xi = torch.from_numpy(x)[None].requires_grad_(True)
    y = m(xi)
    (y[0, 0, C, C] + 0 * y.sum()).backward(retain_graph=True)    # U at centre
    gu = xi.grad[0].abs().numpy().copy(); xi.grad = None
    y[0, 1, C, C].backward()                                     # V at centre
    gv = xi.grad[0].abs().numpy()
    return gu + gv                                               # (3, 256, 256)


res = {}
for cs in ('0.0', '0.2'):
    g = sensitivity(cs)
    tot = g.sum(0); w = tot / tot.sum()
    res[cs] = dict(g=g, frame=g.sum((1, 2)) / g.sum(),
                   cum={R: w[r < R].sum() for R in (4, 8, 16, 32, 64, 128)},
                   extent=np.argwhere(tot > 0))
    e = res[cs]['extent']
    print(f'stage {cs}cs: nonzero rows {e[:,0].min()}-{e[:,0].max()} cols {e[:,1].min()}-{e[:,1].max()} | '
          f'frame share {np.round(res[cs]["frame"], 3)} | ' +
          ' '.join(f'r<{R}:{v:.2f}' for R, v in res[cs]['cum'].items()))

INK, MUTED = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK})
fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True,
                        gridspec_kw={'width_ratios': [1, 1, 1, 1.15]})
g0 = res['0.0']['g']
vmax = g0.max(); norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
for k in range(3):
    ax = axs[k]
    im = ax.imshow(g0[k] + 1e-30, origin='lower', cmap='Blues', norm=norm)
    ax.plot(C, C, '+', color='#eb6834', ms=12, mew=2)
    for R in (16, 64):
        ax.add_patch(plt.Circle((C, C), R, fill=False, color=MUTED, lw=0.8, ls='--'))
    ax.set_title(f'Stage 0: |∂(U,V)[centre] / ∂x[{k}]|   ({100 * res["0.0"]["frame"][k]:.0f}% of total)')
    ax.set_xlabel('column (px)'); ax.set_ylabel('row (px)')
fig.colorbar(im, ax=axs[2], shrink=0.85, label='sensitivity (log scale)')

ax = axs[3]
Rs = np.arange(1, 129)
for cs, col, lab in (('0.0', '#2a78d6', 'stage 0 (0.0cs)'), ('0.2', '#eb6834', 'stage 1 (0.2cs)')):
    tot = res[cs]['g'].sum(0); w = tot / tot.sum()
    ax.plot(Rs, [w[r < R].sum() for R in Rs], color=col, lw=2, label=lab)
ax.set_xscale('log'); ax.set_ylim(0, 1); ax.grid(True, color='#e4e3df', lw=0.6)
ax.set_xlabel('radius from output pixel (px)'); ax.set_ylabel('cumulative share of sensitivity')
ax.set_title('How far the model looks'); ax.legend(frameon=False, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Which input pixels affect the predicted velocity at the centre pixel (step-2 sample, '
             'orange +; dashed circles r = 16, 64 px)', color=INK, fontsize=11, fontweight='bold')
fig.savefig(OUT, dpi=110)
print('saved', OUT)
