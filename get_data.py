import numpy as np
import matplotlib.pyplot as plt
from nm_lib import nm_lib as nm
from matplotlib.animation import FuncAnimation

### Get vid for 1D cases

def animation1D(xx, tt, ut, rhot, Pgt, nt, title, label_x=['x'], figsize=(12, 4)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    def init():
        axes[0].plot(xx, Pgt[:, 0], label='P', color='green')
        axes[1].plot(xx, rhot[:, 0], label='rho', color='blue')
        axes[2].plot(xx, ut[:, 0], label='u', color='red')
        axes[0].set_ylim(np.min(Pgt)-0.1, np.max(Pgt)+0.1)
        axes[1].set_ylim(np.min(rhot)-0.1, np.max(rhot)+0.1)
        axes[2].set_ylim(np.min(ut)-0.1, np.max(ut)+0.1)
        for k in range(3):
            axes[k].set_xlabel(label_x[0])
            axes[k].legend()

    def animate(i):
        for k in range(3):
            axes[k].clear()
        axes[0].plot(xx, Pgt[:, i], label='P', color='green')
        axes[1].plot(xx, rhot[:, i], label='rho', color='blue')
        axes[2].plot(xx, ut[:, i], label='u', color='red')
        axes[0].set_ylim(np.min(Pgt)-0.1, np.max(Pgt)+0.1)
        axes[1].set_ylim(np.min(rhot)-0.1, np.max(rhot)+0.1)
        axes[2].set_ylim(np.min(ut)-0.1, np.max(ut)+0.1)
        for k in range(3):
            axes[k].set_xlabel(label_x[0])
            axes[k].legend()
            axes[k].set_title(f't={tt[i]:.2f}')

    anim = FuncAnimation(fig, animate, interval=50, frames=nt, init_func=init)
    anim.save(f'{title}.mp4', writer='ffmpeg')


def animation_2D(tt, data, nt, label, title):
    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                                'height_ratios': [1]})
    im = None

    def init2d():
        nonlocal im
        im = axes[0].imshow(data[:, :, 0], extent=(0, 1, 0, 1))
        cbar = plt.colorbar(im, ax=axes[1], fraction=1, pad=0.04)

    def animate2D(i):
        nonlocal im
        im.set_array(data[:, :, i])

        axes[0].set_title(f'{label} | t={tt[i]:.2f}')

    anim = FuncAnimation(fig, animate2D, interval=50, frames=nt, init_func=init2d)
    anim.save(f'{title}_{label}.mp4', writer='ffmpeg')

def gauss(xx, A, mu, sigma, C):
    ans = A*np.exp(-.5*(xx - mu)**2/sigma**2) + C
    return ans

def gauss2D(xx, yy, A, mu, sigma, C):
    x_term = -.5*(xx - mu[0])**2/sigma[0]**2
    y_term = -.5*(yy - mu[1])**2/sigma[1]**2
    ans = A*np.exp(x_term + y_term) + C
    return ans

# Same for everything
# Sod test could be a function, but time is lacking
gamma = 5/3
nt = 1000
nt_sod = 200
#x_sod, P_sod, rho_sod, u_sod, t_sod = nm.solve_sod_array(0.2, nt)
#nm.animate_sod(x_sod, P_sod, rho_sod, u_sod, t_sod, 'Sod analytical')
# Different below
data = []
# For x case
nx, ny, nz = 1024, 1, 1
nx2 = int(nx/2)
idx = (slice(None), 0, 0)
idx_sod_L = (slice(nx2, None), 0, 0)
idx_sod_R = (slice(None, nx2), 0, 0)

data.append([nx, ny, nz, idx, idx_sod_L, idx_sod_R])
# For y case
nx, ny, nz = 1, 1024, 1
nx2 = int(ny/2)
idx = (0, slice(None), 0)
idx_sod_R = (0, slice(nx2, None), 0)
idx_sod_L = (0, slice(None, nx2), 0)
data.append([nx, ny, nz, idx, idx_sod_L, idx_sod_R])

# For z case
nx, ny, nz = 1, 1, 1024
nx2 = int(nz/2)
idx = (0, 0, slice(None))
idx_sod_R = (0, 0, slice(nx2, None))
idx_sod_L = (0, 0, slice(None, nx2))
data.append([nx, ny, nz, idx, idx_sod_L, idx_sod_R])

for i in range(3):
    #method = ['FTCS', 'LAX']
    method = ['LAX']
    nx, ny, nz, idx, idx_sod_L, idx_sod_R = data[i]
    for met in method:

        domain = (nx, ny, nz)
        xx = nm.spatial_domain(nx, x0=0, xf=1)
        yy = nm.spatial_domain(ny, x0=0, xf=1)
        zz = nm.spatial_domain(nz, x0=0, xf=1)
        ux0 = np.zeros((nx, ny, nz))
        uy0 = np.zeros((nx, ny, nz))
        uz0 = np.zeros((nx, ny, nz))
        if nx > 1:
            spatial = xx
            dim = 'x'
        elif ny > 1:
            spatial = yy
            dim = 'y'
        elif nz > 1:
            spatial = zz
            dim = 'z'

        print(f'{dim}')

        ux0[idx] = 0.01
        e0 = np.zeros((domain))
        e0[idx] = gauss(spatial, 1, 0.5, sigma=0.01, C=0.1)
        Pg0 = (gamma - 1)*e0
        rho0 = np.zeros((domain))
        rho0[idx] = 1.

        Pg, rho, momentx, momenty, momentz, e, time = nm.solver(domain, xx, yy, zz, nt, Pg0, rho0, ux0, uy0, uz0, e0, method=met)
        if nx > 1:
            moment = momentx
        elif ny > 1:
            moment = momenty
        elif nz > 1:
            moment = momentz
        u = moment/rho
        animation1D(spatial, time, u[idx], rho[idx], Pg[idx], nt, f'{dim} dim {met}', label_x=dim)

        PL = 1.
        PR = 0.1
        ux = np.zeros((domain))
        uy = np.zeros((domain))
        uz = np.zeros((domain))

        if nx > 1:
            ux[idx] = 0.01
        elif ny > 1:
            uy[idx] = 0.01
        elif nz > 1:
            uz[idx] = 0.01

        e0 = np.zeros((domain))
        e0[idx_sod_L] = PL/(gamma - 1)
        e0[idx_sod_R] = PR/(gamma - 1)
        Pg0 = (gamma - 1)*e0

        rho0 = np.zeros((domain))
        rho0[idx_sod_L] = 1.
        rho0[idx_sod_R] = 0.125

        Pg, rho, momentx, momenty, momentz, e, time = nm.solver(domain, xx, yy, zz, nt, Pg0, rho0, ux0, uy0, uz0, e0, method=met)
        if nx > 1:
            moment = momentx
        elif ny > 1:
            moment = momenty
        elif nz > 1:
            moment = momentz
        u = moment/rho
        animation1D(spatial, time, u[idx], rho[idx], Pg[idx], nt, f'Sod numerical {dim} with {met}', label_x=dim)
"""
method = "LAX"
nt = 1000
# 2D time XY plane
title = 'XY'
nx, ny, nz = 256, 256, 1
domain = (nx, ny, nz)
idx = (slice(None), slice(None), 0)
# for YZ plane
"""
"""
title = 'YZ'
nx, ny, nz = 1, 128, 128
idx = (0, slice(None), slice(None))
data.append([nx, ny, nz, idx, title])
# for ZX plane
title = 'ZX'
nx, ny, nz = 128, 1, 128
idx = (slice(None), 0, slice(None))
data.append([nx, ny, nz, idx, title])
"""
"""
print('2D')
print(f'{title}')

xx = nm.spatial_domain(nx, x0=0, xf=1)
yy = nm.spatial_domain(ny, x0=0, xf=1)
zz = nm.spatial_domain(nz, x0=0, xf=1)

X, Y = np.meshgrid(xx, yy)
ux0 = np.zeros((domain))
uy0 = np.zeros((domain))
uz0 = np.zeros((domain))

ux0 = 1.
uy0 = 1.

Pg0 = np.zeros((domain))
Pg0[:, :, 0] = 1.
e0 = Pg0/(gamma - 1)

rho0 = np.zeros((domain))
print(rho0.shape)
rho0[:, :, 0] = gauss2D(X, Y, 1, [0.5, 0.5], sigma=[0.1, 0.1], C=0.1)

Pg, rho, momentx, momenty, momentz, e, time = nm.solver(domain, xx, yy, zz, nt, Pg0, rho0, ux0, uy0, uz0, e0, method='LAX')
ux = momentx/rho
uy = momenty/rho
uz = momentz/rho
u = np.sqrt(ux*ux + uy*uy + uz*uz)
animation_2D(time, rho[:, :, 0], nt, label='rho', title=f'Gaussian rho in {title}')
animation_2D(time, e[:, :, 0], nt, label='e', title=f'Gaussian rho in {title}')
animation_2D(time, u[:, :, 0], nt, label='u', title=f'Gaussian rho in {title}')


# 2D dropping test
print('2D dropping')
print(f'{title}')

ux0 = np.zeros((domain))
uy0 = np.zeros((domain))
uz0 = np.zeros((domain))

e0 = np.zeros((domain))
e0[:, :, 0] = gauss2D(X, Y, 1, [0.5, 0.5], sigma=[0.1, 0.1], C=0.1)
Pg0 = (gamma - 1)*e0

rho0 = np.zeros((domain))
rho0[:, :, 0] = 1.

Pg, rho, momentx, momenty, momentz, e, time = nm.solver(domain, xx, yy, zz, nt, Pg0, rho0, ux0, uy0, uz0, e0, method=method)
ux = momentx/rho
uy = momenty/rho
uz = momentz/rho
u = np.sqrt(ux*ux + uy*uy + uz*uz)

animation_2D(time, rho[:, :, 0], nt, label='rho', title=f'Gaussian e in {title}')
animation_2D(time, e[:, :, 0], nt, label='e', title=f'Gaussian e in {title}')
animation_2D(time, u[:, :, 0], nt, label='u', title=f'Gaussian e in {title}')

"""
