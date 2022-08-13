from SI_Toolkit.Functions.TF import Compile

@Compile
def runge_kutta(self, x, Q, h):
    k1 = self.cartpole_ode(x, Q)
    k2 = self.cartpole_ode(x + 0.5 * k1 * h, Q)
    k3 = self.cartpole_ode(x + 0.5 * k2 * h, Q)
    k4 = self.cartpole_ode(x + k3 * h, Q)

    return x + k1 * h / 6 + k2 * h / 3 + k3 * h / 3 + k4 * h / 6