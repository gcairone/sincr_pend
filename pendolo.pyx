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
cdef double g = -9.80665

# classe pendolo
cdef class Pendolo:

    # angolo iniziale
    cdef public double theta_0
    # velocità angolare iniziale
    cdef public double omega_0
    # vettore con gli angoli per ogni tempo
    cdef public cnp.ndarray theta
    # vettore con le velocità angolari per ogni tempo
    cdef public cnp.ndarray omega
    # vettore con l'energia potenziale (grevitazionale) per ogni tempo
    cdef public cnp.ndarray U
    # vettore con l'energia cinetica (calcolata nel sistema non inerziale del pendolo) per ogni tempo
    cdef public cnp.ndarray K

    def __init__(self, double thetaIniz, double omegaIniz=0):
        self.theta_0 = thetaIniz # pos. iniziale in rad
        self.omega_0 = omegaIniz  # vel. iniziale in rad/s
        self.theta = np.zeros(1)  # array con pos.
        self.omega = np.zeros(1)  # array con vel.
        self.U = np.zeros(1)  # array con U
        self.K = np.zeros(1)  # array con K

    def e_p(self, double m, double l) -> cnp.ndarray:
        # calcola l'energia potenziale considerando il punto di oscillazione del pendolo come U = 0
        # NOTA: si calcola l'energia dopo aver aggiunto tutti gli angoli al vettore self.theta

        # inizializzo il vettore dei risultati a 0
        cdef cnp.ndarray res = np.zeros_like(self.theta)

        # per ogni angolo nel tempo si calcola l'energia potenziale
        for index, elem in enumerate(self.theta):
            res[index] = -m * g * l * cos(elem)
        return res

    def e_c(self, double m, double l, cnp.ndarray vel):
        # calcola l'energia cinetica considerando la velocità relativa al sistema di riferimento del sostegno
        # NOTA: si calcola l'energia cinetica dopo aver aggiunto i risultati a self.theta e self.omega

        # per ogni tempo si calcola l'energia cinetica
        cdef cnp.ndarray res = np.zeros_like(self.omega)
        for index, elem in enumerate(self.omega):
            res[index] = .5*m*(l**2 * elem**2 + vel[index]**2 + 2*l*elem*vel[index]*cos(self.theta[index]))
        return res



cdef class Sistema:
    
    # lista dei pendoli presenti nel sistema
    cdef public list pendoli
    # massa del supporto
    cdef public double M
    # massa dei pendoli 
    cdef public double m
    # lunghezza dei bracci dei pendoli
    cdef public double l
    # numero dei pendoli del sistema
    cdef public int N 
    # numero di intervalli di tempo
    cdef public int n 
    # vettore delle posizioni del cm del supporto per ogni tempo
    cdef public cnp.ndarray x
    # vettore delle velocità del cm del supporto per ogni tempo
    cdef public cnp.ndarray v
    # vettore dell'energia cinetica totale del sistema per ogni tempo
    cdef public cnp.ndarray K
    # vettore dell'energia meccanica totale del sistema per ogni tempo
    cdef public cnp.ndarray E
    # vettore che contiene un linspace che rappresenta il tempo 
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
        self.K = np.zeros(1, dtype=DTYPE)
        self.E = np.zeros(1, dtype=DTYPE)

    def simula(self, double time, double timestep, int damping=1, double mi=0.3, double teta0=PI / 12):
        # calcola il numero di intervalli di tempo
        self.n = int(time / timestep)
        # inizializza in tempo in modo che abbia n + 1 intervalli tra il tempo iniziale e finale
        self.t = np.linspace(0, time, self.n + 1, dtype=DTYPE)
        # inizializza K e E a 0
        self.K = np.zeros_like(self.t)
        self.E = np.zeros_like(self.t)

        # inizializza le condizionni iniziali a 0
        cdef cnp.ndarray initial_cond = np.zeros(2 * self.N + 2, dtype=DTYPE)
        #  VECCHIA VERSIONE
        # for index, p in enumerate(self.pendoli):
        #     initial_cond[index] = p.theta_0
        #     initial_cond[index] = p.omega_0

        # assegna le condizioni iniziali al vettore che contiene le condizioni iniziali
        for i in range(self.N):
            initial_cond[2 * i] = self.pendoli[i].theta_0
            initial_cond[2 * i + 2] = self.pendoli[i].omega_0
        # x e v iniziali del supporto sono già inizializzate a 0

        @cython.boundscheck(False) 
        @cython.wraparound(False)
        def dSdt(cnp.ndarray S):
            # S   di tipo [ang1, ome1, ..., angk, omek, pos, vel]
            # res di tipo [ome1, acc1, ..., omek, acck, vel, acc]

            # inizializzo il vettore dei risultati a 0
            cdef cnp.ndarray res = np.zeros(2 * self.N + 2, dtype=DTYPE)

            
            # calcolo l'accelerazione del supporto 
            cdef double acc = self.m *  sum(self.l * (S[k + 1] ** 2 * sin(S[k])) - g * sin(2 * S[k]) for k in range(0, 2 * self.N, 2))
            acc /= self.M + self.m * sum(sin(S[k]) ** 2 for k in range(0, 2 * self.N, 2))


            # calcolo omega e acc ang
            cdef double acc_ang

            for k in range(0, 2 * self.N, 2):
                # sposto le velocità dal vettore iniziale a quello dei risultati 
                res[k] = S[k + 1]

                # calcolo l'acc ang in funzione della velocita e posizione
                acc_ang = (g * sin(S[k]) - acc * cos(S[k])) / self.l

                # controllo se è attivo il damping e effettuo le modifiche necessarie all'accelerazione
                if damping: acc_ang -= mi * S[k + 1] * ((S[k] / teta0) ** 2 - 1)

                # aggiungo l'acc ang ai risultati
                res[k + 1] = acc_ang

            # sposto la velocità del supporto dal vettore iniziale a quello dei risultati
            res[2 * self.N] = S[2 * self.N + 1]
            # aggiungo l'accelerazione al vettore dei risultati
            res[2 * self.N + 1] = acc

            return res
        
        # METODO DI PRINCE - TROPPO LENTO/ OTTIMIZZATO MALE 
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
            # calcola uno step di integrazione date le condizioni iniziali, lo step h e la funzione che regola il moto

            # utilizza il metodo di Runge-Kutta di 4o ordine
            cdef cnp.ndarray k1 = h*fun(yn)
            cdef cnp.ndarray k2 = h*fun(yn + (1/2)*k1)
            cdef cnp.ndarray k3 = h*fun(yn + (1/2)*k2)
            cdef cnp.ndarray k4 = h*fun(yn + k3)

            # calcola la nuova posizione e la restituisce per l'iterazione successiva
            return yn + (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4

        @cython.boundscheck(False) 
        @cython.wraparound(False)
        def integrate_all(fun: callable, y0: cnp.ndarray, double h) -> cnp.ndarray:
            # calcola per ogni istante di tempo l'integrazine in base alle condizioni di partenza

            # iniziallizza il vettore dei risultati a 0
            cdef cnp.ndarray res = np.zeros(shape=(self.n + 1, 2 * self.N + 2), dtype=DTYPE)
            # il primo istante è quello delle condizioni iniziali
            res[0] = y0

            # assegna a il tempo da valutare il tempo iniziale
            t_eval = self.t

            # intera lungo il tempo
            for t, index in zip(t_eval, range(1, len(t_eval))):
                # per ogni istante di tempo aggiunge al risultato l'initegrazione corrispondente che riporta quindi posizione e velocità
                res[index] = integrate_t(fun, res[index - 1], h)
                # print(res[index], end="\n")

            return res

        # calcola la soluzione chiamando la funzione per integrare sull'intervallo di tempo
        cdef cnp.ndarray solution = integrate_all(dSdt, initial_cond, timestep)
        
        # mette le soluzioni nelle giuste variabili
        self.x = solution[:, 2 * self.N]
        self.v = solution[:, 2 * self.N + 1]
        
        j = 0
        for p in self.pendoli:
            p.theta = solution[:,j]
            p.omega = solution[:,j + 1]
            j += 2

        # calcola l'energia potenziale e cinetica dei singoli pendoli
        for p in self.pendoli:
            p.U = p.e_p(self.m, self.l)
            p.K = p.e_c(self.m, self.l, self.v)
        
        # calcola l'energia cinetica del supporto per ogni istante di tempo
        for index, elem in enumerate(self.v):
            self.K[index] = .5 * self.M * elem ** 2
        
        # calcola l'energia meccanica totale per ogni istante di tempo
        for index in range(len(self.t)):
            
            # si aggiungono tutte le energie potenziali e cinetiche dei singoli pendoli
            for p in self.pendoli:
                self.E[index] += p.U[index] + p.K[index]
            
            # si aggiunge l'energia cinetica del supporto 
            self.E[index] += self.K[index]


    def grafici_pend(self):
        fig, axs = plt.subplots(4)

        # plotta i grafici di t-omega, spazio fasi omega-theta, t-x, spazio fasi x-v
        for p in self.pendoli:
            axs[0].plot(self.t, p.theta)
            axs[1].plot(p.theta, p.omega)
        axs[2].plot(self.t, self.x)
        axs[3].plot(self.x, self.v)
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
        plt.savefig('fig1.png', dpi=600)
        return ani

    def diff(self, p1=0, p2=1):
        if p1 < 0 or p2 < 0: return
        if p1 >= len(self.pendoli) or p2 >= len(self.pendoli): return
        dif = self.pendoli[p1].theta - self.pendoli[p2].theta
        plt.plot(self.t, dif)
        plt.show()
