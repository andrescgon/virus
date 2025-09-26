# **Integrantes**
+ Andres Castro
+ Juan Hurtado
+ Franco Comas

# **Virus**

Este proyecto es un **simulador SIR basado en agentes** (ABM) construido con **Mesa 3.x** y una interfaz interactiva en **Bokeh/Panel**. Cada agente representa a una persona que **se mueve en una grilla**, interactúa con otras y **cambia de estado** (Susceptible → Infected → Removed) según reglas de contagio y recuperación. La app permite **encender y combinar intervenciones** (mascarillas, distanciamiento, aislamiento y vacunación inicial) y observar su efecto en la **dinámica del brote**.

+ **¿Qué resuelve?**  
  + Explorar cómo **transmisión, movilidad y densidad** influyen en el **pico de infecciones** y la **tasa de ataque**.  
  + Evaluar el impacto de **medidas no farmacológicas** (mascarillas, distanciamiento, aislamiento) y **vacunación** sobre la curva.

+ **¿Cómo funciona?**  
  + Los agentes se desplazan por celdas y, cuando coinciden, existe una **probabilidad de contagio** (*p_trans*) modulada por mascarillas y distanciamiento.  
  + Los infectados se **recuperan o mueren** tras un tiempo aleatorio (media y desviación ajustables) y pueden **aislarse** tras un retardo de detección.  
  + El modelo registra **nuevos contagios, infectados activos, removidos y muertes** en cada *step*.

+ **¿Qué puedes controlar?**  
  + **Tamaño de población y grilla**, **probabilidad de transmisión**, **movilidad**, **mortalidad por paso**.  
  + **Vacunación inicial**, **cumplimiento/eficacia de mascarillas**, **distanciamiento**, **aislamiento** y **retardo de detección**.

+ **¿Qué se visualiza?**  
  + **Curvas S/I/R** (área y líneas) que se **autoajustan** al tamaño poblacional.  
  + **Composición actual** (barra horizontal) y **mapa de partículas** donde ves moverse a los agentes y sus estados.




## **Requerimientos**

+ **Python**: 3.9 o superior
+ **Paquetes:**
  + `mesa==3.*`
  + `bokeh`
  + `panel`
  + `numpy`
  + `pandas`

## **Cómo usar la interfaz**

La app tiene dos pestañas: Resumen y Espacio.

### Controles

+ Start: inicia la simulación.
+ Stop: pausa la simulación.
+ Reset: reinicia el modelo con los valores actuales de los sliders.
+ Aplicar parámetros (Reset): aplica los sliders y reinicia (equivale a cambiar sliders y pulsar Reset).


## **Sliders (parámetros)**

| Slider                           | Qué controla                                                            | Efecto práctico                                                                                                   |
| -------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Población**                    | Número total de agentes iniciales.                                      | Aumenta densidad de la grilla → más contactos potenciales. Las gráficas **S/I/R** ajustan sus ejes dinámicamente. |
| **Ancho grid** / **Alto grid**   | Tamaño de la grilla (torus).                                            | Grillas más grandes → menor densidad (si la población no cambia) → menos contactos.                               |
| **p_trans (base)**               | Probabilidad base de contagio al contacto.                              | Aumenta/disminuye la velocidad de propagación. Se ve en la pendiente de **Infected**.                             |
| **Mortalidad por paso**          | Probabilidad de morir por agente infectado en cada paso.                | Agentes fallecidos se eliminan del sistema (no aparecen como R). Aumenta **Muertes** en métricas.                 |
| **Recuperación media**           | Días/steps promedio de recuperación.                                    | Recuperación más lenta mantiene **Infected** elevado por más tiempo.                                              |
| **Recuperación sd**              | Desviación estándar de recuperación.                                    | Introduce heterogeneidad en la duración de infecciones.                                                           |
| **Vacunación inicial**           | Fracción de población que inicia **Removed** (inmune).                  | Menor población susceptible disponible → menor pico y/o brote más corto.                                          |
| **Cumplimiento mascarilla**      | Probabilidad de que un agente use mascarilla.                           | Reduce la transmisión efectiva en contactos, combinada con **Eficacia mascarilla**.                               |
| **Eficacia mascarilla**          | Reducción relativa de transmisión si el emisor/receptor usa mascarilla. | Eficacias más altas reducen significativamente contagios.                                                         |
| **Cumplimiento distanciamiento** | Probabilidad de que un agente practique distanciamiento.                | Agentes cumplidores evitan ciertos contactos, reduciendo incidentes de contagio.                                  |
| **Efecto distanciamiento**       | Probabilidad con la que un distanciador evita el contacto.              | A mayor valor, mayor evitación de contactos.                                                                      |
| **Movilidad (0–1)**              | Probabilidad de moverse cada paso (1 = movimiento normal).              | Menor movilidad reduce mezclado y contagios; 0 ≈ confinamiento.                                                   |
| **Cumplimiento aislamiento**     | Probabilidad de que un infectado se aísle tras ser “detectado”.         | Reduce contagios una vez alcanzado el **Retardo detección**.                                                      |
| **Retardo detección (steps)**    | Steps desde infección hasta que un cumplidor inicia aislamiento.        | Aislamiento temprano reduce la ventana de contagio de cada infectado.                                             |

## **Visualizaciones**

+ **Resumen**
  + **S/I/R — Área apilada**
    + Muestra cantidades **Susceptible (S)**, **Infected (I)** y **Removed (R)** acumuladas por *step*.
    + El eje **Y** se autoajusta al tamaño de la población.
    + El eje **X** se **extiende dinámicamente** en tiempo real cuando la simulación supera el rango inicial.
  <img width="807" height="390" alt="image" src="https://github.com/user-attachments/assets/98ec5831-50aa-4b16-8ae1-cc3486f16dde" />


  + **S/I/R (líneas)**
    + Mismas series, pero en **curvas individuales** para S, I y R (útil para ver cruces y pendientes).
  + **Composición actual (barra horizontal)**
    + Conteo en el *step* actual de **S**, **I** y **R**.
  <img width="801" height="367" alt="image" src="https://github.com/user-attachments/assets/fee77240-8539-47d9-983b-230a49238383" />

  + **Métricas**
    + **Pico de Infectados:** máximo de la curva **I**.
    + **Ataque final (%):** fracción que terminó en **R** respecto a la población inicial.
    + **Rt promedio (proxy):** aproximación basada en **nuevas infecciones por infectado previo** (indicativa).
    + **Muertes:** conteo acumulado de fallecimientos.
  <img width="382" height="341" alt="image" src="https://github.com/user-attachments/assets/57ad28fc-21e8-4a2e-bb3b-82f1d98b0730" />


+ **Espacio**
  + **Mapa de partículas (agentes)**
    + Cada agente se dibuja con *jitter* dentro de su celda.
    + **Colores:**
      + **Azul** = Susceptible
      + **Naranja** = Infected
      + **Verde** = Removed
    + El mapa **se actualiza en cada step** para mostrar **movimiento**, **colisiones** y **transiciones de estado**.
  <img width="587" height="644" alt="image" src="https://github.com/user-attachments/assets/18538700-1fdc-4c4d-aed2-19a07c713424" />

