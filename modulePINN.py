import deepxde as dde
from deepxde.backend import tf
import numpy as np
import matplotlib.pyplot as plt

# Clase para geometría
class Geometry():
  def __init__(self, mode="barra"):
    self.mode = mode # Tipo de geometría

  def geometry_domain(self, params):
    
    timedomain = dde.geometry.TimeDomain(params[0], params[1])

    if self.mode == "barra":
      # Se define un dominio espacio-temporal para la barra
      geom = dde.geometry.Interval(params[2], params[3])

    if self.mode == "placa":
      # Se define un dominio espacio-temporal para la placa
      geom = dde.geometry.Rectangle(params[2], params[3])

    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    return geom, timedomain, geomtime

# Clase para condiciones iniciales
class IC_BC():
    def __init__(self, geomtime):
        self.geomtime = geomtime

    def problem_conditions(self, value, f):
      
      ic = dde.icbc.IC(self.geomtime, f, lambda _, on_initial: on_initial)

      bc = dde.icbc.DirichletBC(self.geomtime, lambda x: value, lambda _, on_boundary:on_boundary)
      
      return ic, bc


# Clase para generar datos sintéticos
class loadData():
  def __init__(self, pde, geomtime, ic, bc):
    self.pde = pde
    self.geomtime = geomtime
    self.ic = ic
    self.bc = bc

  def get_data(self, ndom, nbound, ninit, ntest):
    
    data = dde.data.TimePDE(
      self.geomtime,
      self.pde,
      [self.bc, self.ic],
      num_domain=ndom,
      num_boundary=nbound,
      num_initial=ninit,
      num_test=ntest
      )
      
    return data


# Clase para construir la red
class PINN():
  def __init__(self, n_input, n_hidden, n_output, activation, mmm):

    self.n_input = n_input
    self.n_hidden = n_hidden
    self.n_output = n_output
    self.activation = activation
    self.mmm = mmm

    self.net = dde.nn.FNN([self.n_input] + [self.n_hidden] + [self.n_output], self.activation, self.mmm)

  def train_model(self, data, steps):
    model = dde.Model(data, self.net)
    model.compile("adam", lr=1e-3)

    losshistory, train_state = model.train(iterations=steps)

    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()

    return model, losshistory, train_state


# Clase para graficación
class Ploter():
  def __init__(self, size):
    self.size = size

  def collocation(self, data, var):
    plt.figure(figsize=self.size)
    plt.scatter(data.train_x[:,0], data.train_x[:,1], s=2)
    plt.xlabel('x'); plt.ylabel(var)
    plt.title("Puntos de colocación")
    plt.show()

  def train_plot(self, losshistory, train_state):
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

  def surface_plot(self, grid_terms, mode="barra"):
    if mode == "barra":

      T, X, U_pred = grid_terms

      fig = plt.figure(figsize=self.size)

      ax2 = fig.add_subplot(1, 2, 2, projection='3d')
      surf = ax2.plot_surface(T, X, U_pred, cmap='jet', edgecolor='none', alpha=0.8)
      ax2.set_xlabel('x')
      ax2.set_ylabel('t')
      ax2.set_zlabel('u(x, t)')
      ax2.set_title('u(x, t) PINN')
      fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=12)
      plt.tight_layout()
      plt.show()

    if mode == "placa":

      X, Y, T, U_pred, k = grid_terms
      unique_times = np.unique(T)

      fig = plt.figure(figsize=self.size)
      ax3d = fig.add_subplot(111, projection="3d")
      ax3d.plot_surface(
          X[:, :, k],
          Y[:, :, k],
          U_pred[:, :, k],
          cmap="jet",
          rstride=1,
          cstride=1,
          linewidth=0,
          antialiased=False,
          vmin=0,
          vmax=1
      )
      ax3d.set_xlabel("x")
      ax3d.set_ylabel("y")
      ax3d.set_zlabel("u")
      ax3d.set_title(f"Superficie u(x, y, t={unique_times[k]:.3f})")
      plt.tight_layout()
      plt.show()

  def heatmap_plot(self, grid_terms, mode="barra"):
    if mode == "barra":

      fig= plt.figure(figsize=self.size)

      ax1 = fig.add_subplot(1, 2, 1)
      pcm = ax1.pcolormesh(grid_terms[0], grid_terms[1], grid_terms[2], shading='auto', cmap='jet')
      ax1.invert_yaxis()
      ax1.set_xlabel('t')
      ax1.set_ylabel('x')
      ax1.set_title('Heatmap: u(x, t) PINN')
      fig.colorbar(pcm, ax=ax1, label='u(x, t)')
      plt.tight_layout()
      plt.show()

    if mode == "placa":

      X, Y, T, U_pred, k = grid_terms

      U_final = U_pred[:, :, k]

      unique_times = np.unique(T)

      # Mapa de calor final
      plt.figure(figsize=(6, 4))
      plt.imshow(
          U_final.T,
          origin="lower",
          extent=[0, 1, 0, 1],
          cmap="jet",
          vmin=0,
          vmax=1
      )
      plt.colorbar(label=f"u(x, y, t={unique_times[k]:.3f})")
      plt.xlabel("x")
      plt.ylabel("y")
      plt.title(f"Solución PINN 2D en t = {unique_times[k]:.3f}")
      plt.tight_layout()
      plt.show()


# Clase para construir los meshgrid
class Makegrid():
  def __init__(self, mode="2d"):
    self.mode = mode

  def grid_2D(self, n, params, model):
    if self.mode == "2d":

      t = np.linspace(params[0], params[1], n)
      x = np.linspace(params[2], params[3], n)

      T, X = np.meshgrid(t, x)
      T_flat = T.flatten()[:, None]
      X_flat = X.flatten()[:, None]
      XT = np.hstack((X_flat, T_flat))

      u_pred_flat = model.predict(XT)
      U_pred = u_pred_flat.reshape(n, n)

      return T, X, U_pred

  def grid_3D(self, n, k, params, model):
    if self.mode == "3d":
      t = np.linspace(params[0], params[1], n)
      x = np.linspace(params[2][0], params[3][0], n)
      y = np.linspace(params[2][1], params[3][1], n)

      X, Y, T = np.meshgrid(x, y, t, indexing="xy")

      # Reshape para predict
      XT = np.vstack([
          X.ravel(), Y.ravel(), T.ravel()
      ]).T

      # Obtener predicción de DeepXDE
      u_pred = model.predict(XT)[:, 0]
      U_pred = u_pred.reshape(n, n, n)

      return X, Y, T, U_pred, k