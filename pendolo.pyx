import cython
import numpy as np
cimport numpy as cnp
import matplotlib.pyplot as plt
import matplotlib.animation as amt
from scipy import integrate
from libc.math cimport sin
from libc.math cimport cos

cnp.import_array()

DTYPE = np.float64

ctypedef cnp.float64_t DTYPE_t

cdef double PI = 3.141592653589793238462

# classe pendolo
cdef class Pendolo:

    cdef public double theta_0
    cdef public double omega_0
    cdef public cnp.ndarray theta
    cdef public cnp.ndarray omega

    def __init__(self, double thetaIniz, double omegaIniz=0):
        self.theta_0 = thetaIniz # pos. iniziale in rad
        self.omega_0 = omegaIniz  # vel. iniziale in rad/s
        self.theta = np.zeros(1)  # array con pos.
        self.omega = np.zeros(1)  # array con vel.


cdef class Sistema:

    cdef public list pendoli
    cdef public double M 
    cdef public double m
    cdef public double l
    cdef public int N 
    cdef public int n 
    cdef public cnp.ndarray x
    cdef public cnp.ndarray v
    cdef public cnp.ndarray t

    def __init__(self, pend: list, double Mass, double mass, double lenght):
        self.pendoli = pend  # lista di pendoli
        self.M = Mass  # massa supporto
        self.m = mass  # massa su ciascun pendolo
        self.l = lenght  # lunghezza dei pendoli
        self.N = len(self.pendoli) # numero di pendoli
        self.x = np.zeros(1, dtype=DTYPE)  # pos. supporto
        self.v = np.zeros(1, dtype=DTYPE)  # vel. supporto
        self.t = np.zeros(1, dtype=DTYPE)  # tempo

    def simula(self, double time, double timestep, int damping=1, double mi=0.3, double teta0=PI / 12):
        cdef double g = -9.81
        self.n = int(time / timestep)
        self.t = np.linspace(0, time, self.n + 1, dtype=DTYPE)

        # lista di cond iniz
        cdef cnp.ndarray initial_cond = np.zeros(2 * self.N + 2, dtype=DTYPE)
        # for index, p in enumerate(self.pendoli):
        #     initial_cond[index] = p.theta_0
        #     initial_cond[index] = p.omega_0

        for i in range(self.N):
            initial_cond[2 * i] = self.pendoli[i].theta_0
            initial_cond[2 * i + 2] = self.pendoli[i].omega_0
        # pos e vel iniz del supporto sono inizializzate a zero 

        @cython.boundscheck(False) 
        @cython.wraparound(False)
        def dSdt(cnp.ndarray S):
            # S   di tipo [ang1, ome1, ..., angk, omek, pos, vel]
            # res di tipo [ome1, acc1, ..., omek, acck, vel, acc]
            cdef cnp.ndarray res = np.zeros(2 * self.N + 2, dtype=DTYPE)

            cdef double acc = self.m *  sum(self.l * (S[k + 1] ** 2 * sin(S[k])) - g * sin(2 * S[k]) for k in range(0, 2 * self.N, 2))
            acc /= self.M + self.m * sum(sin(S[k]) ** 2 for k in range(0, 2 * self.N, 2))
            # calcolo omega e acc ang
            cdef double acc_ang
            for k in range(0, 2 * self.N, 2):
                res[k] = S[k + 1]
                acc_ang = (g * sin(S[k]) - acc * cos(S[k])) / self.l
                if damping: acc_ang -= mi * S[k + 1] * ((S[k] / teta0) ** 2 - 1)
                res[k + 1] = acc_ang
            res[2 * self.N] = S[2 * self.N]
            res[2 * self.N + 1] = S[2 * self.N + 1]
            return res
        
        
        # def integrate_t(fun: callable, cnp.ndarray yn , float h):
        #     cdef cnp.ndarray k1 = h*fun(yn)
        #     cdef cnp.ndarray k2 = h*fun(yn + (1/5)*k1 )
        #     cdef cnp.ndarray k3 = h*fun(yn + (3/40)*k1 + (9/40)*k2)
        #     cdef cnp.ndarray k4 = h*fun(yn + (44/45)*k1 - (56/15)*k2 + (39/9)*k3)
        #     cdef cnp.ndarray k5 = h*fun(yn + (19372/6561)*k1 - (25360/2187)*k2 + (64448/6561)*k3 - (212/729)*k4)
        #     cdef cnp.ndarray k6 = h*fun(yn + (9017/3168)*k1 - (355/33)*k2 - (46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5)
        #     # k7 = h*fun(yn + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6)

        #     return yn + (35/384)*k1 + (500/1113)*k3 + (135/192)*k4 - (2187/6784)*k5 + (11/84)*k6 

        def integrate_t(fun: callable, cnp.ndarray yn , float h):
            cdef cnp.ndarray k1 = h*fun(yn)
            cdef cnp.ndarray k2 = h*fun(yn + (1/2)*k1)
            cdef cnp.ndarray k3 = h*fun(yn + (1/2)*k2)
            cdef cnp.ndarray k4 = h*fun(yn + k3)
            return yn + (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4

        @cython.boundscheck(False) 
        @cython.wraparound(False)
        def integrate_all(fun: callable, y0: cnp.ndarray, double h) -> cnp.ndarray:
            cdef cnp.ndarray res = np.zeros(shape=(self.n + 1, 2 * self.N + 2), dtype=DTYPE)
            res[0] = y0
            t_eval = self.t
            for t, index in zip(t_eval, range(1, len(t_eval))):
                res[index] = integrate_t(fun, res[index - 1], h)
                #print(res[index], end="\n")
            return res
        
        #solution = integrate.solve_ivp(dSdt, t_span=[0, self.t[-1]], y0=initial_cond, t_eval=self.t)
        solution = integrate_all(dSdt, initial_cond, timestep)
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
                    [self.l * (i + sin(self.pendoli[i].theta[j * speed])) - (
                            self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                        k + sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (self.M + nn * self.m),
                     # x del pendolo
                     self.l * i - (self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                         k + sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (self.M + nn * self.m)],
                    # x del perno
                    [self.l * 1.5 - self.l * cos(self.pendoli[i].theta[j * speed]),  # y del pendolo
                     self.l * 1.5])  # y del perno
                i += 1
            i = 0
            for lin in vert:
                # aste verticali
                lin.set_data(
                    [self.l * i - (self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                        k + sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (self.M + nn * self.m),
                     # x del perno
                     self.l * i - (self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                         k + sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (self.M + nn * self.m)],
                    # x del perno
                    [self.l * 1.5,  # y del perno
                     0])  # y del perno - altezza
                i += 1
            base.set_data([- self.l * 0.5 - (self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                k + sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (self.M + nn * self.m),
                           self.l * (nn - 0.5) - (self.M * self.l * (nn - 1) * 0.5 + self.m * self.l * sum(
                               k + sin(self.pendoli[k].theta[j * speed]) for k in range(nn))) / (
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
