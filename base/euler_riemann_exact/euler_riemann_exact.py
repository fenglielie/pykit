import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from typing import Optional, Tuple, Dict


def euler_riemann_exact_solver(
    *,
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    gamma: float = 1.4,
    xlist: Optional[np.ndarray] = None,
    x_c: float = 0.0,
    t: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    # Calculate sound speed in left and right states
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)

    # Calculate constants alpha and beta
    alpha = (gamma + 1.0) / (gamma - 1.0)
    beta = (gamma - 1.0) / (2.0 * gamma)

    # Check for cavitation (solution for cavitation not supported)
    if u_l - u_r + 2 * (c_l + c_r) / (gamma - 1.0) < 0:
        raise ValueError("Cavitation detected!  Exiting.")

    # Define integral curves and Hugoniot locus for 1-wave and 3-wave
    def integral_curve_1(p: float) -> float:
        return u_l + 2 * c_l / (gamma - 1.0) * (1.0 - (p / p_l) ** beta)

    def integral_curve_3(p: float) -> float:
        return u_r - 2 * c_r / (gamma - 1.0) * (1.0 - (p / p_r) ** beta)

    def hugoniot_locus_1(p: float) -> float:
        return u_l + 2 * c_l / np.sqrt(2 * gamma * (gamma - 1.0)) * (
            (1 - p / p_l) / np.sqrt(1 + alpha * p / p_l)
        )

    def hugoniot_locus_3(p: float) -> float:
        return u_r - 2 * c_r / np.sqrt(2 * gamma * (gamma - 1.0)) * (
            (1 - p / p_r) / np.sqrt(1 + alpha * p / p_r)
        )

    def phi_l(p: float) -> float:
        return hugoniot_locus_1(p) if p >= p_l else integral_curve_1(p)

    def phi_r(p: float) -> float:
        return hugoniot_locus_3(p) if p >= p_r else integral_curve_3(p)

    # Construct the intersection equation in the (p-v) plane
    func = lambda p: phi_l(p) - phi_r(p)
    # Initial guess p0
    p0_PV = (p_l + p_r) / 2.0 - 1 / 8 * (u_r - u_l) * (rho_l + rho_r) * (c_l + c_r)
    p0 = np.max([p0_PV, 1e-8])

    # Solve for the intersection point to get the intermediate state p and u
    p_s_tmp, _, ier, msg = opt.fsolve(func, p0, full_output=True, xtol=1.0e-12)
    p_s = p_s_tmp.item()
    u_s = 0.5 * (phi_l(p_s) + phi_r(p_s))

    # Warning if the solution fails to converge
    if ier != 1:
        print("Warning: fsolve did not converge.", msg)

    # Calculate the density on the left and right side of the contact discontinuity

    if p_s <= p_l:
        rho_s_l = (p_s / p_l) ** (1.0 / gamma) * rho_l  # Rarefaction wave
    else:
        rho_s_l = (
            (1.0 + alpha * p_s / p_l) / ((p_s / p_l) + alpha)
        ) * rho_l  # Shock wave

    if p_s <= p_r:
        rho_s_r = (p_s / p_r) ** (1.0 / gamma) * rho_r  # Rarefaction wave
    else:
        rho_s_r = (
            (1.0 + alpha * p_s / p_r) / ((p_s / p_r) + alpha)
        ) * rho_r  # Shock wave

    # Calculate sound speed in the intermediate state
    c_s_l = np.sqrt(gamma * p_s / rho_s_l)
    c_s_r = np.sqrt(gamma * p_s / rho_s_r)

    # Calculate wave speeds

    # 1-wave
    if p_s > p_l:  # Shock wave
        w_1_l = (rho_l * u_l - rho_s_l * u_s) / (rho_l - rho_s_l)
        w_1_r = w_1_l
    else:  # Rarefaction wave
        w_1_l = u_l - c_l
        w_1_r = u_s - c_s_l

    # 2-wave
    w_2 = u_s

    # 3-wave
    if p_s > p_r:  # Shock wave
        w_3_l = (rho_r * u_r - rho_s_r * u_s) / (rho_r - rho_s_r)
        w_3_r = w_3_l
    else:  # Rarefaction wave
        w_3_l = u_s + c_s_r
        w_3_r = u_r + c_r

    w_max = max(np.abs([w_1_l, w_1_r, w_2, w_3_l, w_3_r]))

    # Automatically choose appropriate spatial and temporal scales if parameters are missing
    if xlist is None:
        xlist = np.linspace(-1.0, 1.0, 1000)
        x_c = 0.0
    if t is None:
        t = 0.8 * max(np.abs(xlist - x_c)) / w_max

    # Warning if the wave is out of range
    if t * w_max > max(np.abs(xlist - x_c)):
        print("Warning: wave is out of range.")

    # Solve for the state inside the rarefaction wave
    xi = (xlist - x_c) / t  # type: ignore
    u_1_fan = ((gamma - 1.0) * u_l + 2 * (c_l + xi)) / (gamma + 1.0)
    u_3_fan = ((gamma - 1.0) * u_r - 2 * (c_r - xi)) / (gamma + 1.0)
    rho_1_fan = (rho_l**gamma * (u_1_fan - xi) ** 2 / (gamma * p_l)) ** (
        1.0 / (gamma - 1.0)
    )
    rho_3_fan = (rho_r**gamma * (xi - u_3_fan) ** 2 / (gamma * p_r)) ** (
        1.0 / (gamma - 1.0)
    )
    p_1_fan = p_l * (rho_1_fan / rho_l) ** gamma
    p_3_fan = p_r * (rho_3_fan / rho_r) ** gamma

    # Calculate return values

    rho_out = np.zeros_like(xlist)
    u_out = np.zeros_like(xlist)
    p_out = np.zeros_like(xlist)

    for i, xi_val in enumerate(xi):
        if xi_val <= w_1_l:  # Left of the 1-wave
            rho_out[i], u_out[i], p_out[i] = rho_l, u_l, p_l
        elif xi_val <= w_1_r:  # Inside the 1-wave (if it's a rarefaction wave)
            rho_out[i], u_out[i], p_out[i] = rho_1_fan[i], u_1_fan[i], p_1_fan[i]
        elif xi_val <= w_2:  # Between the 1-wave and the 2-wave
            rho_out[i], u_out[i], p_out[i] = rho_s_l, u_s, p_s
        elif xi_val <= w_3_l:  # Between the 2-wave and the 3-wave
            rho_out[i], u_out[i], p_out[i] = rho_s_r, u_s, p_s
        elif xi_val <= w_3_r:  # Inside the 3-wave (if it's a rarefaction wave)
            rho_out[i], u_out[i], p_out[i] = rho_3_fan[i], u_3_fan[i], p_3_fan[i]
        else:  # Right of the 3-wave
            rho_out[i], u_out[i], p_out[i] = rho_r, u_r, p_r

    # Provide more detailed information if requested

    more_info = {
        "p_s": p_s,
        "u_s": u_s,
        "rho_s_l": rho_s_l,
        "rho_s_r": rho_s_r,
        "w_1_l": w_1_l,
        "w_1_r": w_1_r,
        "w_2": w_2,
        "w_3_l": w_3_l,
        "w_3_r": w_3_r,
    }

    left_type = "Shock" if p_s > p_l else "Rarefaction"
    center_type = "Contract discontinuity"
    right_type = "Shock" if p_s > p_r else "Rarefaction"

    more_info["left_type"] = left_type
    more_info["center_type"] = center_type
    more_info["right_type"] = right_type

    more_info["type_msg"] = f"[L] {left_type}\n[C] {center_type}\n[R] {right_type}"
    more_info["key_msg"] = f"{p_s=}\t{u_s=}\t{rho_s_l=}\t{rho_s_l=}"

    return rho_out, u_out, p_out, more_info


def euler_riemann_exact_plot(
    *,
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    x_l: float,
    x_r: float,
    t: float,
    x_c: float = 0.0,
):
    xlist = np.linspace(x_l, x_r, 1000)
    rho, u, p, more_info = euler_riemann_exact_solver(
        rho_l=rho_l,
        u_l=u_l,
        p_l=p_l,
        rho_r=rho_r,
        u_r=u_r,
        p_r=p_r,
        xlist=xlist,
        x_c=x_c,
        t=t,
    )

    gamma = 1.4
    e = np.zeros_like(p)
    for i in range(len(e)):
        e[i] = p[i] / ((gamma - 1) * rho[i])

    fig = plt.figure(figsize=(8, 8))
    primitive = [rho, u, p, e]
    names = ["Density", "Velocity", "Pressure", "Internal Energy"]
    for i in range(4):
        axe = fig.add_subplot(2, 2, i + 1)
        q = primitive[i]
        plt.plot(xlist, q, linewidth=2)
        plt.title(names[i])
        qmax = max(q)
        qmin = min(q)
        qdiff = qmax - qmin
        axe.set_ylim((qmin - 0.1 * qdiff, qmax + 0.1 * qdiff))

    fig, ax = plt.subplots()
    tlist = np.linspace(0, t, 500)

    if more_info["left_type"] == "Shock":  # Left shock
        w_1_line = x_c + more_info["w_1_l"] * tlist
        ax.plot(w_1_line, tlist, "red", label="Left shock")
    else:  # Left Rarefaction
        w_1_l_line = x_c + more_info["w_1_l"] * tlist
        w_1_r_line = x_c + more_info["w_1_r"] * tlist
        ax.fill_betweenx(
            tlist,
            w_1_l_line,
            w_1_r_line,
            color="red",
            alpha=0.2,
            label="Left Rarefaction",
        )

    # Contact discontinuity in the middle
    w_2_line = x_c + more_info["w_2"] * tlist
    ax.plot(w_2_line, tlist, "green", label="Contract discontinuity")

    if more_info["right_type"] == "Shock":  # Right shock
        w_3_line = x_c + more_info["w_3_l"] * tlist
        ax.plot(w_3_line, tlist, "blue", label="Right shock")
    else:  # Right rarefaction
        w_3_l_line = x_c + more_info["w_3_l"] * tlist
        w_3_r_line = x_c + more_info["w_3_r"] * tlist
        ax.fill_betweenx(
            tlist,
            w_3_l_line,
            w_3_r_line,
            color="blue",
            alpha=0.2,
            label="Right rarefaction",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("data", x_c))
    ax.spines["bottom"].set_position(("data", 0))
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_label_position("bottom")
    ax.yaxis.set_label_position("left")
    ax.set_xticks([x_l, x_c, x_r])
    ax.set_yticks([t])
    ax.set_yticklabels(["$t$"])
    ax.set_xlim(x_l, x_r)
    ax.set_ylim(0, t)
    ax.legend()

    fig.suptitle(f"Riemann problem (t={t})")

    return fig, ax


def euler_riemann_density_plot(
    rho_l: float,
    u_l: float,
    p_l: float,
    rho_r: float,
    u_r: float,
    p_r: float,
    x_l: float,
    x_r: float,
    t: float,
    x_c: float = 0.0,
):
    time_N = 100
    time_dt = t / time_N
    xlist = np.linspace(x_l, x_r, 800)

    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["blue", "cyan", "green", "yellow", "red"]
    )

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_title("Density Distribution")
    ax.set_xlabel("Position")
    ax.set_yticks([])
    ax.set_xticks([x_l, x_c, x_r])

    time_text = ax.text(0.02, 0.9, "", transform=ax.transAxes)

    rho = np.where(xlist < x_c, rho_l, rho_r)
    rho_1, _, _, _ = euler_riemann_exact_solver(
        rho_l=rho_l,
        u_l=u_l,
        p_l=p_l,
        rho_r=rho_r,
        u_r=u_r,
        p_r=p_r,
        xlist=xlist,
        x_c=x_c,
        t=10*time_dt,
    )
    norm = Normalize(vmin=np.min(rho_1), vmax=np.max(rho_1))
    density_matrix = np.tile(rho, (4, 1))

    cax = ax.imshow(
        density_matrix, aspect="auto", cmap=cmap, extent=(x_l, x_r, 0, 1), norm=norm
    )
    fig.colorbar(cax, ax=ax, orientation="vertical")

    def update(frame):
        t_now = frame * time_dt
        if t_now == 0:
            rho = np.where(xlist < x_c, rho_l, rho_r)
        else:
            rho, _, _, _ = euler_riemann_exact_solver(
                rho_l=rho_l,
                u_l=u_l,
                p_l=p_l,
                rho_r=rho_r,
                u_r=u_r,
                p_r=p_r,
                xlist=xlist,
                x_c=x_c,
                t=t_now,
            )
        density_matrix = np.tile(rho, (30, 1))
        cax.set_data(density_matrix)
        time_text.set_text(f"T={t_now:.2g}")
        return (cax,)

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=time_N,
        interval=20,
    )

    return ani
