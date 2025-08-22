import deepxde as dde
from deepxde.geometry import Cuboid, CSGUnion  # O Box si Cuboid no existe
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deepxde.backend import tf

# Parámetros geométricos
base_width = 0.1
base_height = 3.0
base_depth = 0.05
fin_length = 0.9
fin_thickness = 0.05
fin_depth = base_depth
num_fins = 7
fin_spacing = base_height / (num_fins + 1)

# Crear geometría 3D
base = Cuboid([0.0, 0.0, 0.0], [base_width, base_height, base_depth])
geom = base
for i in range(num_fins):
    y_pos = fin_spacing * (i + 1) - fin_thickness / 2
    fin = Cuboid(
        [base_width, y_pos, 0.0],
        [base_width + fin_length, y_pos + fin_thickness, fin_depth]
    )
    geom = CSGUnion(geom, fin)

# Parámetros físicos
a = 0.4  # Difusión
beta = 0.4  # Convección

# PDE estacionaria: a * ∇²u - beta * u = 0
def pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    du_zz = dde.grad.hessian(u, x, i=2, j=2)
    laplacian = du_xx + du_yy + du_zz
    return a * laplacian - beta * u  # Residual

# BC Dirichlet: Base caliente en x=0
def boundary_base(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

bc_base = dde.DirichletBC(geom, lambda x: 1.0, boundary_base)  # u=1

# BC Robin corregido: Convección en aletas
h_over_k = 1.0  # Coeficiente convectivo
def boundary_fins(x, on_boundary):
    return on_boundary and (x[0] >= base_width)  # Caras expuestas en aletas

bc_convection = dde.RobinBC(
    geom,
    lambda x, outputs: -h_over_k * outputs,  # -α * u
    boundary_fins
)

bcs = [bc_base, bc_convection]

# Datos para PDE (estacionario, sin tiempo)
num_domain = 1000  # Interiores
num_boundary = 2000  # Fronteras
num_test = 2000
data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=num_domain,
    num_boundary=num_boundary,
    num_test=num_test,
    train_distribution="Hammersley"
)

# Red neuronal: Inputs 3D (x,y,z)
net = dde.nn.FNN([3] + [50] + [1], "tanh", "Glorot normal")

# Modelo y entrenamiento
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=20000)  # Adam

model.compile("L-BFGS")
losshistory, train_state = model.train()  # Refinamiento

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Predicción: Muestrear puntos y predecir u
pred_points = geom.uniform_boundary_points(500000)  # Interiores
#pred_points = geom.random_points(500000)  # Interiores
u_pred = model.predict(pred_points)

# Visualización 3D de temperaturas
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], c=u_pred.flatten(), cmap='jet', s=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Distribución Estacionaria de Temperaturas en el Disipador')
#ax.view_init(azim=5, elev=30, roll=90)
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Temperatura u')
ax.view_init(azim=45, elev=30)
plt.show()


######################### RRR ##########################


import deepxde as dde
from deepxde.geometry import Cuboid, CSGUnion
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deepxde.backend import tf
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Parámetros geométricos
base_width = 0.1
base_height = 3.0
base_depth = 0.05
fin_length = 0.9
fin_thickness = 0.05
fin_depth = base_depth
num_fins = 6
fin_spacing = base_height / (num_fins + 1)

# Crear geometría espacial 3D
base = Cuboid([0.0, 0.0, 0.0], [base_width, base_height, base_depth])
geom_space = base
for i in range(num_fins):
    y_pos = fin_spacing * (i + 1) - fin_thickness / 2
    fin = Cuboid(
        [base_width, y_pos, 0.0],
        [base_width + fin_length, y_pos + fin_thickness, fin_depth]
    )
    geom_space = CSGUnion(geom_space, fin)

# Dominio temporal (t de 0 a 1)
timedomain = dde.geometry.TimeDomain(0.0, 1.0)

# Geometría espacio-temporal
geom = dde.geometry.GeometryXTime(geom_space, timedomain)

# Parámetros físicos
a = 0.4  # Difusión
beta = 0.4  # Convección

# PDE transitoria: ∂u/∂t = a * ∇²u - beta * u
def pde(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=3)  # ∂u/∂t (x[:,3] es t)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    du_zz = dde.grad.hessian(u, x, i=2, j=2)
    laplacian = du_xx + du_yy + du_zz
    return du_t - a * laplacian + beta * u  # Residual ajustado

# BC Dirichlet: Base caliente en x=0
def boundary_base(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

bc_base = dde.icbc.DirichletBC(geom, lambda x: 1.0, boundary_base)  # u=1

# BC Robin corregido: Convección en aletas
h_over_k = 1.0  # Coeficiente convectivo
def boundary_fins(x, on_boundary):
    return on_boundary and (x[0] >= base_width)  # Caras expuestas en aletas

bc_convection = dde.icbc.RobinBC(
    geom,
    lambda x, outputs: -h_over_k * outputs,  # -α * u
    boundary_fins
)

bcs = [bc_base, bc_convection]

# IC: Temperatura inicial u=0 (frío)
ic = dde.icbc.IC(geom, lambda x: 0.0, lambda _, on_initial: on_initial)

# Datos para TimePDE (no estacionario, con tiempo)
num_domain = 2000  # Interiores
num_boundary = 2000  # Fronteras
num_initial = 600  # Puntos iniciales
num_test = 2000
data = dde.data.TimePDE(
    geom,
    pde,
    bcs + [ic],
    num_domain=num_domain,
    num_boundary=num_boundary,
    num_initial=num_initial,
    num_test=num_test,
    train_distribution="Hammersley"
)

# Red neuronal: Inputs 4D (x,y,z,t)
net = dde.nn.FNN([4] + [50] * 4 + [1], "tanh", "Glorot normal")

# Modelo y entrenamiento
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=10000)  # Adam

model.compile("L-BFGS")
losshistory, train_state = model.train()  # Refinamiento

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Predicción para animación: Muestrear puntos espaciales fijos y variar t
# Reducimos a 10000 puntos para mejorar rendimiento (original era 1M, excesivo para animación)
pred_points = geom_space.random_points(500000)
num_time_steps = 50  # Frames en la animación
t_values = np.linspace(0.0, 2.0, num_time_steps)

# Precomputar u para cada t (en batches para memoria)
batch_size_pred = 2000  # Ajustado para evitar OOM en Colab
u_pred_all = []
for t in t_values:
    t_array = np.full((pred_points.shape[0], 1), t)
    X_pred = np.hstack((pred_points, t_array))  # [x,y,z,t]
    u_pred_t = []
    for i in range(0, len(X_pred), batch_size_pred):
        u_pred_t.append(model.predict(X_pred[i:i + batch_size_pred]))
    u_pred_t = np.concatenate(u_pred_t, axis=0)
    u_pred_all.append(u_pred_t.flatten())

# Animación 3D del flujo de calor (evolución de temperaturas)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter([], [], [], c=[], cmap='jet', s=2)  # Inicializar scatter vacío
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Distribución de Temperaturas en el Disipador (t=0.0)')
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Temperatura u')
ax.view_init(azim=45, elev=30)
ax.set_xlim(0, base_width + fin_length)
ax.set_ylim(0, base_height)
ax.set_zlim(0, base_depth)

def update(frame):
    scatter._offsets3d = (pred_points[:, 0], pred_points[:, 1], pred_points[:, 2])
    scatter.set_array(u_pred_all[frame])
    ax.set_title(f'Distribución de Temperaturas en el Disipador (t={t_values[frame]:.2f})')
    return scatter,

# Crear y mostrar animación con HTML en Colab
ani = FuncAnimation(fig, update, frames=num_time_steps, interval=100, blit=True)
HTML(ani.to_jshtml())