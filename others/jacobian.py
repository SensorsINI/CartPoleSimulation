import sympy as sym
from sympy.utilities.lambdify import lambdify, implemented_function


def _cartpole_ode(angle, angleD, positionD, u):
    """Should be the same function as in CartPole.cartpole_model,
    except that sympy trigonometric functions are used here
    """
    ca = sym.cos(-angle)
    sa = sym.sin(-angle)

    A = m * (ca ** 2) - (k + 1) * (M + m)

    positionDD = (
        (
            + m * g * sa * ca
            - ((J_fric * (-angleD) * ca) / L)
            - (k + 1) * (
                + (m * L * (angleD ** 2) * sa)
                - M_fric * positionD
                + u
            )
        ) / A
    )

    angleDD = (
        (
            g * sa - positionDD * ca - (J_fric * (-angleD)) / (m * L) 
        ) / ((k + 1) * L)
    ) * (-1.0)

    return angleDD, positionDD


x, v, t, o, u = sym.symbols("x,v,t,o,u")
k, M, m, L, J_fric, M_fric, g = sym.symbols("k,M,m,L,J_fric,M_fric,g")

xD = v
tD = o
oD, vD = _cartpole_ode(t, o, v, u)


xx = sym.diff(xD, x, 1)
xv = sym.diff(xD, v, 1)
xt = sym.diff(xD, t, 1)
xo = sym.diff(xD, o, 1)
xu = sym.diff(xD, u, 1)

vx = sym.diff(vD, x, 1)
vv = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(vD, v, 1), "numpy")
vt = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(vD, t, 1), "numpy")
vo = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(vD, o, 1), "numpy")
vu = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(vD, u, 1), "numpy")

tx = sym.diff(tD, x, 1)
tv = sym.diff(tD, v, 1)
tt = sym.diff(tD, t, 1)
to = sym.diff(tD, o, 1)
tu = sym.diff(tD, u, 1)

ox = sym.diff(oD, x, 1)
ov = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(oD, v, 1), "numpy")
ot = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(oD, t, 1), "numpy")
oo = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(oD, o, 1), "numpy")
ou = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(oD, u, 1), "numpy")



