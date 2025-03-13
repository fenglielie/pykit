import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable, Tuple


def burgers_sin_exact_solver_kernel(x: np.ndarray, t: float, ep: float) -> np.ndarray:
    """
    u(x,t): exact solution of burgers equation when u0(x) = sin(x)
    """

    if t < 0:
        raise ValueError(f"{t=}<0")
    if ep <= 0:
        ep = 1e-6

    # x in [-pi, pi]
    x = np.mod(x + np.pi, 2 * np.pi) - np.pi

    # The choice of initial value ​​in Newton's method is very sensitive!
    k = 1.0 / (np.pi / 2 + t)
    u = k * x

    iter_max: int = 100000
    iter: int = 0
    while iter < iter_max:
        tmp = 1 + np.cos(x - u * t) * t
        du = (u - np.sin(x - u * t)) / tmp
        u = u - du

        iter += 1
        if np.max(np.abs(du)) < ep:
            break

    return u


def burgers_sin_exact_solver(
    x: np.ndarray, t: float, a: float, b: float, w: float, phi: float, ep: float
) -> np.ndarray:
    """
    u(x,t): exact solution of burgers equation when u0(x) = a + b sin(w x + phi)
    """

    return a + b * burgers_sin_exact_solver_kernel(
        w * x + phi - a * w * t, b * w * t, ep
    )


def burgers_sin_shock(
    xleft: float, xright: float, t: float, a: float, b: float, w: float, phi: float
) -> Tuple[float, np.ndarray]:
    """
    Return the earliest time of shock wave appearance and shock positions.

    Tb: Earliest time of shock wave appearance.
    y: Shock positions within the range [xleft, xright].
    """

    tb = np.abs(1 / (b * w))

    if t >= tb:
        y0 = a * t + (np.pi - phi) / w
        tx = 2 * np.pi / w

        y0 = np.mod((y0 - xleft), tx) + xleft
        y = np.arange(y0, xright, tx)
    else:
        y = np.array([])

    return tb, y


def burgers_sin_show(
    xleft: float,
    xright: float,
    xnum: int,
    t: float,
    a: float,
    b: float,
    w: float,
    phi: float,
    ep: float,
    retFigure: bool = False,
):
    x = np.linspace(xleft, xright, xnum)
    u = burgers_sin_exact_solver(x, t, a, b, w, phi, ep)

    fig, axes = plt.subplots()
    tb, y = burgers_sin_shock(xleft, xright, t, a, b, w, phi)

    print(f"u0(x)={a}+{b}*sin({w}*x+{phi})")

    if t >= tb:
        print(f"t({t}) >= tb({tb})")

        if len(y) > 0:
            print(f"Shock wave location: {y}")

            y2 = np.linspace(a - b, a + b, xnum)
            for item in y:
                x2 = item * np.ones(xnum)
                axes.plot(x2, y2, color="r", linestyle="--")
    else:
        print(f"tb = {tb}")

    axes.plot(x, u)

    if retFigure:
        return fig, axes


def burgers_show_line(
    u0: Callable[[np.ndarray], np.ndarray],
    xleft: float,
    xright: float,
    num: int,
    tend: float,
):
    xi_list = np.linspace(xleft, xright, num)
    x_list = np.zeros(num)
    u_list = np.zeros(num)

    fig, axes = plt.subplots()

    axes.plot(xi_list, u0(xi_list), "b", alpha=0.2)

    for i in range(num):
        x = xi_list[i] + u0(xi_list[i]) * tend
        u = u0(xi_list[i])

        xs = np.linspace(xi_list[i], x, num)
        ys = u + np.linspace(0, tend, num)

        x_list[i] = x
        u_list[i] = u + tend
        axes.plot(xs, ys, alpha=0.6)

    axes.plot(x_list, u_list, "r")


def burgers_show_animate(
    u0: Callable[[np.ndarray], np.ndarray],
    xleft: float,
    xright: float,
    num: int,
    tend: float,
) -> animation.FuncAnimation:
    xi_lists = np.linspace(xleft, xright, num)
    u_lists = u0(xi_lists)
    time_N = 100
    time_dt = tend / time_N

    fig, axes = plt.subplots()

    (line,) = axes.plot(xi_lists, u_lists)
    time_text = axes.text(0.02, 0.95, "", transform=axes.transAxes)

    def update(frame: int):
        x_lists = xi_lists + u0(xi_lists) * frame * time_dt
        line.set_data(x_lists, u_lists)
        time_text.set_text(f"T={frame * time_dt:.2f}")
        return line, time_text

    ani = animation.FuncAnimation(fig=fig, func=update, frames=time_N, interval=20)
    return ani
