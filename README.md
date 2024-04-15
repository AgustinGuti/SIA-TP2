# TP2 SIA - Algoritmos Geneticos

Trabajo práctico para la materia Sistemas de Inteligencia Artificial del ITBA con el objetivo de implementar un motor de algoritmos genéticos y ajustar los parámetros para obtener las mejores configuraciones de personajes de un juego de rol. 

Alumnos: Agustín Andrés Gutiérrez, Ian Bernasconi, Jeremías Demián Feferovich y Lucas Miguel Biolley Calvo.

## Set up
Para ejecutar el motor de busquedas:
1. Instalación de librerias
 - ```pip install -r ./requirements.txt```
 - Si se desea comprimir y descomprimir los archivos de resultados automaticamente con cada `merge` y `pull` respectivamente, ejecutar `init.py` en su lugar.

2. Configuración de ejecución. Archivo `config.yaml`
    - Se pueden especificar las variables con las que se desea ejecutar el algoritmo.

3. Configuración de recopilación de datos. Archivos dentro de `run_configs`
    - Dentro de esta carpeta se pueden crear archivos yaml de configuración para ejecutar varias iteraciones con diferentes combinaciones de variables. 
    - Ya se encuentran creados algunos que utilizamos para la extracción de los datos en `data.zip`.
    - Si todos los archivos tienen `active` en `False`, se ejecuta solo una vez, tomando en cuenta el `config.yaml`

## Ejecución
- Para correr el código, ejecutar `python ./engine.py`
- Para correr el análisis de resultados, ejecutar `python ./results.py`. Para configurar que resultados analizar y mostrar, modificar el archivo `graphs_config.yaml`