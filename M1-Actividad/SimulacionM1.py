"""
Descripción: Este programa simula la limpieza de una habitación con múltiples agentes robots y celdas sucias.
Autores: 
Sandra Paulina Herrera Rebolledo  A01798452 
Fernanda Ponce Maciel  A01799293
Fecha de creación/modificación: 08/11/2024
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import seaborn as sns

class RobotLimpieza(Agent):
    'Representa un robot de limpieza que se desplaza y limpia celdas.'

    def __init__(self, modelo):
        super().__init__(modelo.next_id(), modelo)
        self.pasosRealizados = 0

    def step(self):
        '''
        Realiza la acción del robot en un paso de la simulación.
        Si la celda actual está sucia, la limpia. De lo contrario, se mueve a una celda vecina aleatoria.
        '''
        celdaLimpia = False
        for obj in self.model.grid.iter_cell_list_contents(self.pos):
            if isinstance(obj, CeldaSucia):
                self.model.grid.remove_agent(obj)
                self.model.schedule.remove(obj)  # Eliminar la celda sucia del schedule
                print(f"Robot {self.unique_id} limpió una celda en {self.pos}")
                celdaLimpia = True
                break
        
        if not celdaLimpia:
            movimientosPosibles = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            siguienteMovimiento = self.random.choice(movimientosPosibles)
            
            if all(not isinstance(a, RobotLimpieza) for a in self.model.grid.iter_cell_list_contents(siguienteMovimiento)):
                self.model.grid.move_agent(self, siguienteMovimiento)
                self.pasosRealizados += 1
                print(f"Robot {self.unique_id} se movió a {siguienteMovimiento}")

class CeldaSucia(Agent):
    'Representa una celda sucia.'

    def __init__(self, modelo):
        super().__init__(modelo.next_id(), modelo)

    def step(self):
        'Acción de la celda sucia. No realiza ninguna acción.'
        pass

class SimulacionLimpieza(Model):
    'Modelo de simulación de limpieza con múltiples robots y celdas sucias. '

    def __init__(self, ancho=20, alto=20, numRobots=5, porcentajeSuciedad=0.4, pasosMaximos=100):
        '''
        Inicializa el modelo de simulación con el tamaño de la cuadrícula, número de robots,
        porcentaje de celdas sucias y número máximo de pasos.
        '''
        super().__init__()
        self.grid = MultiGrid(ancho, alto, torus=False)
        self.schedule = RandomActivation(self)
        self.pasosMaximos = pasosMaximos
        self.pasoActual = 0

        # Colocar todos los robots en la celda [1, 1]
        for _ in range(numRobots):
            robot = RobotLimpieza(self)
            self.grid.place_agent(robot, (1, 1))
            self.schedule.add(robot)

        # Inicializar celdas sucias en ubicaciones aleatorias
        for (contenido, coordenadas) in self.grid.coord_iter():
            x, y = coordenadas
            if random.random() < porcentajeSuciedad and self.grid.is_cell_empty((x, y)):
                suciedad = CeldaSucia(self)
                self.grid.place_agent(suciedad, (x, y))
                self.schedule.add(suciedad)

        # Configurar la recolección de datos
        self.recolectorDatos = DataCollector(
            model_reporters={
                "porcentajeLimpieza": lambda m: m.calcularPorcentajeLimpieza(),
                "totalMovimientos": lambda m: sum(a.pasosRealizados for a in m.schedule.agents if isinstance(a, RobotLimpieza)),
                "tiempoTranscurrido": lambda m: m.pasoActual
            }
        )

    def calcularPorcentajeLimpieza(self):
        'Calcula el porcentaje de celdas limpias en la cuadrícula.'
        totalCeldas = self.grid.width * self.grid.height
        celdasSucias = sum(1 for a in self.schedule.agents if isinstance(a, CeldaSucia))
        porcentajeLimpio = ((totalCeldas - celdasSucias) / totalCeldas) * 100
        print(f"Paso {self.pasoActual}: {porcentajeLimpio:.2f}% celdas limpias (Celdas sucias: {celdasSucias})")
        return porcentajeLimpio

    def paso(self):
        '''
        Realiza un paso de la simulación, recolectando datos y ejecutando un paso de cada agente.
        Finaliza la simulación si se ha alcanzado el número máximo de pasos o todas las celdas están limpias.
        '''
        self.recolectorDatos.collect(self)
        self.schedule.step()
        self.pasoActual += 1
        self.mostrarCuadricula()
        if self.pasoActual >= self.pasosMaximos or not any(isinstance(a, CeldaSucia) for a in self.schedule.agents):
            self.running = False

    def mostrarCuadricula(self):
        'Visualiza el estado actual de la cuadrícula con colores específicos.'
        mapaCuadricula = np.zeros((self.grid.width, self.grid.height))
        for (contenido, (x, y)) in self.grid.coord_iter():
            for obj in contenido:
                if isinstance(obj, CeldaSucia):
                    mapaCuadricula[x, y] = 2  # Representación de celdas sucias
                elif isinstance(obj, RobotLimpieza):
                    mapaCuadricula[x, y] = 1  # Representación de robots

        cmapPersonalizado = mcolors.ListedColormap(['#add8e6', '#ff69b4', '#8a2be2'])  # Colores personalizados
        plt.imshow(np.flipud(mapaCuadricula), cmap=cmapPersonalizado, vmin=0, vmax=2)  # Invertir el eje y
        plt.title(f"Simulación paso {self.pasoActual}")
        plt.axis('off')
        plt.pause(0.5)
        plt.clf()

# Solicitar al usuario los parámetros de la simulación
ancho = int(input("Ingrese la anchura de la habitación (M): "))
alto = int(input("Ingrese la altura de la habitación (N): "))
numRobots = int(input("Ingrese el número de agentes: "))
porcentajeSuciedad = float(input("Ingrese el porcentaje de celdas inicialmente sucias (0.0 - 1.0): "))
pasosMaximos = int(input("Ingrese el tiempo máximo de ejecución: "))

# Configuración y ejecución de la simulación
modelo = SimulacionLimpieza(ancho, alto, numRobots, porcentajeSuciedad, pasosMaximos)

plt.ion()  # Habilitar modo interactivo para visualizar la simulación paso a paso
for _ in range(pasosMaximos):
    modelo.paso()
plt.ioff()  # Deshabilitar modo interactivo

# Visualización de los resultados
resultados = modelo.recolectorDatos.get_model_vars_dataframe()

sns.set(style="whitegrid")

# Gráfica del porcentaje de celdas limpias respecto al tiempo
plt.figure(figsize=(10, 5))
sns.lineplot(data=resultados, x=resultados.index, y="porcentajeLimpieza", label="Porcentaje de Limpieza")
plt.title("Porcentaje de Celdas Limpias Durante la Simulación")
plt.xlabel("Paso de la Simulación")
plt.ylabel("Porcentaje de Limpieza")
plt.show()

# Gráfica de la sumatoria de movimientos realizados por los agentes
plt.figure(figsize=(10, 5))
sns.lineplot(data=resultados, x=resultados.index, y="totalMovimientos", label="Movimientos Totales")
plt.title("Sumatoria de Movimientos Realizados por los Agentes")
plt.xlabel("Paso de la Simulación")
plt.ylabel("Número de Movimientos")
plt.show()

# Gráfica del tiempo transcurrido
plt.figure(figsize=(10, 5))
sns.lineplot(data=resultados, x=resultados.index, y="tiempoTranscurrido", label="Tiempo Transcurrido")
plt.title("Tiempo Transcurrido Durante la Simulación")
plt.xlabel("Paso de la Simulación")
plt.ylabel("Tiempo Transcurrido")
plt.show()
