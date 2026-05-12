import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Nx = 200
Ny = 200
h = 1
c = 1
dt = 0.5
x_src = 100
y_src = 100
A = 1
omega = 0.5

u_prev = np.zeros((Nx,Ny))
u_curr = np.zeros((Nx,Ny))
u_next = np.zeros((Nx,Ny))
obstacles = np.zeros((Nx,Ny), dtype = bool)

#obstacles[150,:] = True
def run_simulation(u_prev, u_curr, u_next, obstacles, dt, h, c, num_steps, x_src, y_src, A, omega):
    lmbda = (c*dt)/h
    frames = []
    
    for n in range(num_steps):
        laplacian = (u_curr[0:-2, 1:-1] + u_curr[2:, 1:-1] +  u_curr[1:-1, 0:-2] + u_curr[1:-1, 2:] -  4 * u_curr[1:-1, 1:-1])
        u_next[1:-1, 1:-1] = 2 * u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + (lmbda**2) * laplacian
        
        u_next[x_src, y_src] = A * np.sin(omega * n * dt) 
        u_next[obstacles] = 0
    
        u_next[0, :] = 0
        u_next[-1, :] = 0
        u_next[:, 0] = 0
        u_next[:, -1] = 0
    
        if n%5 == 0:
            frames.append(u_curr.copy())

        u_prev = u_curr.copy()
        u_curr = u_next.copy()
    
    return frames

frames = run_simulation(u_prev, u_curr, u_next, obstacles, dt, h, c, num_steps=400, x_src=100, y_src=100, A=1, omega=0.5)

fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap='RdBu', vmin=-1, vmax=1)

def update(frame_data):
    im.set_array(frame_data)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
plt.show() 
