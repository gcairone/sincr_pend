import pendoli
from numpy import pi


# istanziamento dei pendoli, i parametri passati sono posizione iniziale e velocità angolare iniziale (di default = 0)
p1 = pendoli.Pendolo(thetaIniz=pi / 18)
p2 = pendoli.Pendolo(thetaIniz=-pi / 3)
p5 = pendoli.Pendolo(thetaIniz=pi / 5)

# istanziamento supporto, i parametri sono: pendoli da mettere sopra, M, m, l
Sys = pendoli.Sistema(pend=[p1, p2, p5], Mass=5, mass=1, lenght=0.3)
# funzione per simulare l'evoluzione, i parametri sono: durata, timestep, presenza del termine di damping-driving,
# coefficiente miu, angolo di equilibrio
Sys.simula(100, 0.01, damping=True, mi=0.6)
Sys.animazione()


