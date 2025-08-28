from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from IPython.display import HTML


def animation_function_1d(grid_terms):

    T, X, U_pred = grid_terms
    unique_times = np.unique(T)

    # --- Colormap y normalización ---
    norm = Normalize(vmin=U_pred.min(), vmax=U_pred.max())
    cmap = plt.cm.jet

    # --- Figura y ejes ---
    fig, ax = plt.subplots(figsize=(8, 2.5))

    # Eje principal: solo para eje x
    ax.set_xlim(0, 1)
    ax.set_xlabel("Posición en la barra (x)")
    ax.set_yticks([])  # ocultamos ticks de y
    ax.set_title("Difusión del calor en la barra, t = 0.00")

    # Creamos un eje inset, muy delgado, centrado verticalmente
    # width="95%" del ancho del eje principal, height="15%" de su alto
    cax = inset_axes(
        ax,
        width="100%",
        height="5%",
        loc="center",
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cax.set_xticks([])
    cax.set_yticks([])

    # Dibujamos la primera fila como heatmap muy delgado
    im = cax.imshow(
        U_pred[0:1, :],
        cmap=cmap,
        norm=norm,
        aspect="auto",
        extent=[0, 1, 0, 1],  # y va de 0 a 1 dentro del inset (se ve como franja finísima)
    )

    # Añadimos la colorbar fuera del inset, ligada al principal
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("u(x, t)")

    # --- Funciones de animación ---
    def init():
        im.set_data(U_pred[0:1, :])
        ax.set_title(f"t = {unique_times[0]:.2f}")
        return [im]

    def animate(i):
        im.set_data(U_pred[i:i+1, :])
        ax.set_title(f"t = {unique_times[i]:.2f}")
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=100,
        interval=100,
        blit=True,
    )

    # Mostrar en Jupyter Notebook
    anim.save("heat1d_evolution.gif", writer='pillow')
    plt.close(fig)

def animation_function_2d(grid_terms):

    X, Y, T, U_pred, k = grid_terms
    unique_times = np.unique(T)

    U = np.transpose(U_pred, (2, 0, 1))    # ahora U[k], k=0,…,nt−1

    # Establecer figura e imagen inicial para animación
    fig, ax = plt.subplots(figsize=(6, 5))
    vmin, vmax = U.min(), U.max()

    im = ax.imshow(
        U[0],
        origin="lower",
        cmap="jet",
        vmin=0,
        vmax=1,
        animated=True
    )
    cbar = fig.colorbar(im, ax=ax, label=r"$u(x,y,t)$")
    title = ax.set_title(f"t = {unique_times[0]:.3f}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Funciones para init y animación
    def init():
        im.set_data(U[0])
        title.set_text(f"t = {unique_times[0]:.3f}")
        return im, title

    def animate(k):
        im.set_data(U[k])
        title.set_text(f"t = {unique_times[k]:.3f}")
        return im, title

    # Construcción de la animación
    interval_ms = 10*max(20, int(1000 * (unique_times[1] - unique_times[0])))
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=100,
        interval=interval_ms,
        blit=True
    )

    # Para exportar como archivo:
    anim.save("heat2d_evolution.gif", writer='pillow')
    plt.close(fig)