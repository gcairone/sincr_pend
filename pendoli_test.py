import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as amt
from scipy import integrate


# classe pendolo
class Pendolo:
    def __init__(self, thetaIniz, omegaIniz=0):
        self.theta_0 = thetaIniz # pos. iniziale in rad
        self.omega_0 = omegaIniz  # vel. iniziale in rad/s
        self.theta = np.zeros(1)  # array con pos.
        self.omega = np.zeros(1)  # array con vel.


class Sistema:
    def __init__(self, pend, Mass, mass, lenght):
        self.pendoli = pend  # lista di pendoli
        self.M = Mass  # massa supporto
        self.m = mass  # massa su ciascun pendolo
        self.l = lenght  # lunghezza dei pendoli
        self.x = np.zeros(1)  # pos. supporto
        self.v = np.zeros(1)  # vel. supporto
        self.t = np.zeros(1)  # tempo

    def simula(self, time, timestep, damping=True, mi=0.3, teta0=np.pi / 12):
        g = -9.81
        n = int(time / timestep)
        self.t = np.linspace(0, time, n + 1)

        # lista di cond iniz
        initial_cond = []
        for p in self.pendoli:
            initial_cond.append(p.theta_0)
            initial_cond.append(p.omega_0)
        # pos e vel iniz del supporto
        initial_cond.append(0)
        initial_cond.append(0)

        def dSdt(S) -> np.array:
            # S   di tipo [ang1, ome1, ..., angk, omek, pos, vel]
            # res di tipo [ome1, acc1, ..., omek, acck, vel, acc]
            res = []
            # calcolo vel e acc
            vel = S[-1]
            acc = self.m *  sum(
                self.l * (S[k + 1] ** 2 * np.sin(S[k])) - g * np.sin(2 * S[k]) for k in range(0, 2 * len(self.pendoli), 2))
            acc /= self.M + self.m * sum(np.sin(S[k]) ** 2 for k in range(0, 2 * len(self.pendoli), 2))
            # calcolo omega e acc ang
            for k in range(0, 2 * len(self.pendoli), 2):
                res.append(S[k + 1])
                acc_ang = (g * np.sin(S[k]) - acc * np.cos(S[k])) / self.l
                if damping: acc_ang -= mi * S[k + 1] * ((S[k] / teta0) ** 2 - 1)
                res.append(acc_ang)
            res.append(vel)
            res.append(acc)
            return np.array(res)
        
        def integrate_t(fun: callable, yn: np.array, h: float):
            k1 = fun(yn)
            k2 = fun(yn + 0.5*k1*h)
            k3 = fun(yn + 0.5*k2*h)
            k4 = fun(yn + k3*h)

            return yn + (h/6) * (k1 + 2*k3 + 2*k3 + k4)


        def integrate_all(fun: callable, t_span: np.array, y0: np.array, h: float) -> np.array:
            res = np.zeros(shape=(n + 1, 2 * len(self.pendoli) + 2))
            res[0] = y0
            t_eval = self.t
            for t, index in zip(t_eval, range(1, len(t_eval))):
                res[index] = integrate_t(fun, res[index - 1], t)
            return res

        
        #solution = integrate.solve_ivp(dSdt, t_span=[0, self.t[-1]], y0=initial_cond, t_eval=self.t)
        solution = integrate_all(dSdt, [0, self.t[-1]], np.array(initial_cond), timestep)
        # metti le soluzioni nei giusti array
        self.x = solution[:,-2]
        self.v = solution[:,-1]
        j = 0
        for p in self.pendoli:
            p.theta = solution[:,j]
            p.omega = solution[:,j + 1]
            j += 2

    def grafici_pend(self):
        for p in self.pendoli:
            plt.plot(self.t, p.theta)
        plt.show()

    def animazione(self, sp=1):
        # trasformo le coordinate
        # sarebbe la velocita' di esecuzione (1 indica velocita' nomale)
        speed = int(sp * 0.05 / (self.t[1] - self.t[0]))
        n = len(self.t)
        nn = len(self.pendoli)
        # creazione figure
        fig, ax = plt.subplots()
        ax.set_ylim(0, 2 * self.l)
        ax.set_xlim(- self.l * (nn / 2 + 1), self.l * (nn / 2 + 1))
        ax.set_aspect('equal')
        ax.grid()
        # creazione linee
        mobili = []
        vert = []
        base, = ax.plot([], [], '-', lw=10, color='k')
        for p in self.pendoli:
            asta, = ax.plot([], [], '-', lw=2, color='k')
            vert.append(asta)
            linea, = ax.plot([], [], 'o-', lw=2)
            mobili.append(linea)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def animate(j):
            i = 0
            for lin in mobili:
                # aste mobili dei pendoli
                lin.set_data(
                    [self.l * (i + np.sin(self.pendoli[i].theta[j * speed])) - (
                            self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                        k + np.sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (self.M + nn * self.m),
                     # x del pendolo
                     self.l * i - (self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                         k + np.sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (self.M + nn * self.m)],
                    # x del perno
                    [self.l * 1.5 - self.l * np.cos(self.pendoli[i].theta[j * speed]),  # y del pendolo
                     self.l * 1.5])  # y del perno
                i += 1
            i = 0
            for lin in vert:
                # aste verticali
                lin.set_data(
                    [self.l * i - (self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                        k + np.sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (self.M + nn * self.m),
                     # x del perno
                     self.l * i - (self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                         k + np.sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (self.M + nn * self.m)],
                    # x del perno
                    [self.l * 1.5,  # y del perno
                     0])  # y del perno - altezza
                i += 1
            base.set_data([- self.l * 0.5 - (self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                k + np.sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (self.M + nn * self.m),
                           self.l * (nn - 0.5) - (self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                               k + np.sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (
                                   self.M + nn * self.m)],
                          [0, 0])
            time_text.set_text(time_template % (j * speed * (self.t[1] - self.t[0])))
            return mobili, vert

        ani = amt.FuncAnimation(fig, animate, len(self.t), interval=50)
        plt.show()
        return ani

    def diff(self, p1=0, p2=1):
        if p1 < 0 or p2 < 0: return
        if p1 >= len(self.pendoli) or p2 >= len(self.pendoli): return
        dif = self.pendoli[p1].theta - self.pendoli[p2].theta
        plt.plot(self.t, dif)
        plt.show()
