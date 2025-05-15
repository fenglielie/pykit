import numpy as np
import matplotlib.pyplot as plt


def CV_show_line(u0, df, x, tend):
    fig, axes = plt.subplots()
    for xi in x:
        xs = np.linspace(xi, xi + df(u0(xi)) * tend, 5)
        ys = np.linspace(0, tend, 5)

        axes.plot(xs, ys, alpha=0.6)

    axes.set_xlabel("x")
    axes.set_ylabel("t")

    return fig, axes


def HJ_show_line(p0, dH, x, tend):
    return CV_show_line(u0=p0, df=dH, x=x, tend=tend)


def CV_characteristic_equation(*, dF, ddF, u0, du0, solve_xi=True):
    """
    $x = xi + f'(u_0(xi))t$

    $F(xi) = x - xi - f'(u_0(xi))t = 0$, $u(x,t) = u_0(xi)$

    $G(u) = u - u_0(x - f'(u) t) = 0$
    """

    if solve_xi:

        def f_xi(x, t):
            return lambda xi: x - xi - dF(u0(xi)) * t

        def df_xi(x, t):
            return lambda xi: -1 - ddF(u0(xi)) * du0(xi) * t

        def post_xi(xi):
            return u0(xi)

        return f_xi, df_xi, post_xi
    else:

        def f_u(x, t):
            return lambda u: u - u0(x - dF(u) * t)

        def df_u(x, t):
            return lambda u: 1 + du0(x - dF(u) * t) * ddF(u) * t

        def post_u(u):
            return u

        return f_u, df_u, post_u


def HJ_characteristic_equation(*, H, dH, ddH, u0, du0, ddu0):
    """
    $x = xi + H'(p_0(xi)) t$

    $F(xi) = x - xi - H'(p_0(xi)) t = 0$

    $L(p) = p H'(p) - H(p)$

    $phi = phi_0(xi) + L(p_0(xi)) t$
    """

    def L(p):
        return p * dH(p) - H(p)

    def f_xi(x, t):
        return lambda xi: x - xi - dH(du0(xi)) * t

    def df_xi(x, t):
        return lambda xi: -1 - ddH(du0(xi)) * ddu0(xi) * t

    def post_xi(xi, t):
        return u0(xi) + L(du0(xi)) * t

    return f_xi, df_xi, post_xi
