import numpy as np
import deepxde as dde
from deepxde.geometry import Cuboid, CSGUnion
import matplotlib.pyplot as plt
from deepxde.backend import tf
from matplotlib.animation import FuncAnimation

def finGeometry(params):
    
    # Dominio temporal (t de 0 a 1)
    timedomain = dde.geometry.TimeDomain(params[0], params[1])
    
    # Parámetros geométricos
    base_width = params[2] #0.1
    base_height = params[3] #3.0
    base_depth = params[4] #0.05
    fin_length = params[5] #0.9
    fin_thickness = params[6] #0.05
    fin_depth = base_depth
    num_fins = params[7] #6
    fin_spacing = base_height / (num_fins + 1)

    # Crear geometría espacial 3D
    base = Cuboid([0.0, 0.0, 0.0], [base_width, base_height, base_depth])
    geom = base
    for i in range(num_fins):
        y_pos = fin_spacing * (i + 1) - fin_thickness / 2
        fin = Cuboid(
            [base_width, y_pos, 0.0],
            [base_width + fin_length, y_pos + fin_thickness, fin_depth]
        )
        geom = CSGUnion(geom, fin)

    # Geometría espacio-temporal
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    return timedomain, geom, geomtime

def fin_ICBC(geomtime, base_width):
    # BC Dirichlet: Base caliente en x=0
    def boundary_base(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0.0)

    bc_base = dde.icbc.DirichletBC(geomtime, lambda x: 1.0, boundary_base)  # u=1

    # BC Robin corregido: Convección en aletas
    h_over_k = 1.0  # Coeficiente convectivo
    def boundary_fins(x, on_boundary):
        return on_boundary and (x[0] >= base_width)  # Caras expuestas en aletas

    bc_convection = dde.icbc.RobinBC(
        geomtime,
        lambda x, outputs: -h_over_k * outputs,  # -α * u
        boundary_fins
    )

    bcs = [bc_base, bc_convection]

    # IC: Temperatura inicial u=0 (frío)
    ic = dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on_initial: on_initial)

    return bcs, ic

def finData(geomtime, pde, bcs, ic):
    # Datos para TimePDE (no estacionario, con tiempo)
    num_domain = 2000  # Interiores
    num_boundary = 2000  # Fronteras
    num_initial = 600  # Puntos iniciales
    num_test = 2000
    data = dde.data.TimePDE(
        geomtime,
        pde,
        bcs + [ic],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test,
        train_distribution="Hammersley"
    )

    return data

def fin_train_model(data, ninput, nhidden, noutput, steps):
    # Red neuronal: Inputs 4D (x,y,z,t)
    net = dde.nn.FNN([ninput] + [nhidden] + [noutput], "tanh", "Glorot normal")

    # Modelo y entrenamiento
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    losshistory, train_state = model.train(iterations=steps)  # Adam

    model.compile("L-BFGS")
    losshistory, train_state = model.train()  # Refinamiento

    #dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    return model

def finAnimation(geom, model, params):
    # Predicción para animación: Muestrear puntos espaciales fijos y variar t
    pred_points = geom.random_points(500000)  # Shape: (500000, 3) for [x, y, z]
    num_time_steps = 50  # Frames en la animación
    t_values = np.linspace(0.0, 2.0, num_time_steps)

    # Precomputar u para cada t (en batches para memoria)
    batch_size_pred = 2000
    u_pred_all = []
    for t in t_values:
        # Create time array for all points
        t_array = np.full((pred_points.shape[0], 1), t)  # Shape: (500000, 1)
        X_pred = np.hstack((pred_points, t_array))  # Shape: (500000, 4) for [x, y, z, t]
        u_pred_t = []
        for i in range(0, len(X_pred), batch_size_pred):
            batch = X_pred[i:i + batch_size_pred]  # Shape: (batch_size_pred, 4)
            u_pred_t.append(model.predict(batch))
        u_pred_t = np.concatenate(u_pred_t, axis=0)
        u_pred_all.append(u_pred_t.flatten())

    # Animación 3D del flujo de calor
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter([], [], [], c=[], cmap='jet', s=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Distribución de Temperaturas en el Disipador (t=0.0)')
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Temperatura u')
    ax.view_init(azim=45, elev=30)
    ax.set_xlim(0, params[0] + params[1])  # base_width + fin_length
    ax.set_ylim(0, params[2])  # base_height
    ax.set_zlim(0, params[3])  # base_depth

    def update(frame):
        scatter._offsets3d = (pred_points[:, 0], pred_points[:, 1], pred_points[:, 2])
        scatter.set_array(u_pred_all[frame])
        ax.set_title(f'Distribución de Temperaturas en el Disipador (t={t_values[frame]:.2f})')
        return scatter,

    # Crear y mostrar animación
    anim = FuncAnimation(fig, update, frames=num_time_steps, interval=100, blit=True)
    anim.save("heatPipe_evolution.gif", writer='pillow')


# def finAnimation(geom, model, params):
#     # Predicción para animmación: Muestrear puntos espaciales fijos y variar t
#     # Reducimos a 10000 puntos para mejorar rendimiento (original era 1M, excesivo para animmación)
#     pred_points = geom.random_points(500000)
#     num_time_steps = 50  # Frames en la animmación
#     t_values = np.linspace(0.0, 2.0, num_time_steps)

#     # Precomputar u para cada t (en batches para memoria)
#     batch_size_pred = 2000  # Ajustado para evitar OOM en Colab
#     u_pred_all = []
#     for t in t_values:
#         #t_array = np.full((pred_points.shape[0], 1), t)
#         X_pred = np.hstack((pred_points))  # [x,y,z,t]
#         u_pred_t = []
#         for i in range(0, len(X_pred), batch_size_pred):
#             u_pred_t.append(model.predict(X_pred[i:i + batch_size_pred]))
#         u_pred_t = np.concatenate(u_pred_t, axis=0)
#         u_pred_all.append(u_pred_t.flatten())

#     # animmación 3D del flujo de calor (evolución de temperaturas)
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     scatter = ax.scatter([], [], [], c=[], cmap='jet', s=2)  # Inicializar scatter vacío
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_title('Distribución de Temperaturas en el Disipador (t=0.0)')
#     fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Temperatura u')
#     ax.view_init(azim=45, elev=30)
#     ax.set_xlim(0, params[0] + params[1]) #base_width + fin_length
#     ax.set_ylim(0, params[2]) #base_height
#     ax.set_zlim(0, params[3]) #base_depth

#     def update(frame):
#         scatter._offsets3d = (pred_points[:, 0], pred_points[:, 1], pred_points[:, 2])
#         scatter.set_array(u_pred_all[frame])
#         ax.set_title(f'Distribución de Temperaturas en el Disipador (t={t_values[frame]:.2f})')
#         return scatter,

#     # Crear y mostrar animación con HTML en Colab
#     anim = FuncAnimation(fig, update, frames=num_time_steps, interval=100, blit=True)
#     anim.save("heatPipe_evolution.gif", writer='pillow')