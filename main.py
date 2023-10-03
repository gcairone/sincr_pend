import pendolo
import numpy
from numpy import pi
import time


# istanziamento dei pendoli, i parametri passati sono posizione iniziale e velocità angolare iniziale (di default = 0)
p1 = pendolo.Pendolo(thetaIniz=pi / 18)
p2 = pendolo.Pendolo(thetaIniz=-pi / 3)
p5 = pendolo.Pendolo(thetaIniz=pi / 5)

# istanziamento supporto, i parametri sono: pendoli da mettere sopra, M, m, l
Sys = pendolo.Sistema(pend=[p1, p2, p5], Mass=5, mass=1, lenght=0.3)
# funzione per simulare l'evoluzione, i parametri sono: durata, timestep, presenza del termine di damping-driving,
# coefficiente miu, angolo di equilibrio
ti = time.time()
Sys.simula(10, 0.001, damping=0, mi=0.6, useSciPy = 1)
tf = time.time()
print("delta time = ", tf - ti)
#Sys.grafici_pend()
Sys.animazione()


