# ErgoIA

ErgoIA es una aplicacion local que usa la camara de la computadora para revisar tu postura mientras estas sentado.

La app muestra alertas de postura, pausas activas e hidratacion. Es una herramienta educativa; no reemplaza una evaluacion medica o ergonomica profesional.

## 1. Preparar la computadora

Necesitas:

- Windows.
- Python instalado.
- Una camara/webcam conectada.
- Permiso de Windows para que las apps usen la camara.
- Internet la primera vez, para descargar dependencias y modelos.

Para revisar si Python esta instalado, abre PowerShell y ejecuta:

```powershell
python --version
```

Si aparece una version de Python, puedes continuar.

## 2. Abrir la carpeta del proyecto

En PowerShell entra a la carpeta de ErgoIA:

```powershell
cd C:\Users\Jahir\Desktop\IA\ergoia
```

Importante: ejecuta los comandos desde esa carpeta para que `data`, `models` y `datasets` queden en el lugar correcto.

## 3. Crear el entorno virtual

Esto solo se hace una vez:

```powershell
python -m venv .venv
```

Activar el entorno:

```powershell
.\.venv\Scripts\activate
```

Cuando este activo, instala las librerias:

```powershell
pip install -r requirements.txt
```

## 4. Preparar el modelo

Descarga el dataset:

```powershell
python download_datasets.py
```

Entrena el modelo:

```powershell
python train_multiposture_model.py
```

Esto crea el archivo:

```text
models/posture_random_forest.joblib
```

## 5. Ejecutar ErgoIA

Cada vez que quieras usar la app:

```powershell
cd C:\Users\Jahir\Desktop\IA\ergoia
.\.venv\Scripts\activate
python run.py --classifier auto --source 0
```

Si la camara no abre, prueba:

```powershell
python run.py --classifier auto --source 1
python run.py --classifier auto --source 2
```

Tambien puedes abrir la app y usar el boton `Camara` para ver las camaras conectadas y escoger una.

## 6. Como se usa

Al abrirse la ventana:

- Sientate frente a la camara.
- Procura que se vean cabeza, hombros y torso.
- Usa buena iluminacion.
- Mantente dentro del cuadro de video.

La app detecta:

- Postura correcta.
- Cabeza adelantada.
- Espalda encorvada.
- Inclinacion hacia un lado.
- Hombros desalineados.
- Si no hay una persona visible.

## 7. Botones

En la interfaz aparecen estos botones:

- `Salir`: cierra la app.
- `Agua`: registra manualmente que tomaste agua.
- `Pausa`: registra una pausa activa.
- `puntos on/off`: muestra u oculta los puntos del cuerpo.
- `Camara`: muestra las camaras conectadas. Al elegir una, la lista se oculta.

## 8. Teclas rapidas

Tambien puedes usar el teclado:

- `q`: salir.
- `h`: registrar agua.
- `b`: registrar pausa.
- `t`: mostrar u ocultar puntos.
- `c`: mostrar u ocultar lista de camaras.
- `0-9`: elegir una camara por numero.

## 9. Ver historial

Para abrir el panel de historial:

```powershell
cd C:\Users\Jahir\Desktop\IA\ergoia
.\.venv\Scripts\activate
streamlit run app.py
```

Los registros se guardan en:

```text
data/historial_alertas.csv
data/hidratacion.csv
```

## 10. Si algo falla

Si no abre la camara:

- Cierra Zoom, Teams, navegador u otra app que este usando la camara.
- Revisa permisos de camara en Windows.
- Prueba con `--source 0`, `--source 1` o `--source 2`.
- Usa el boton `Camara` dentro de la app.

Si faltan librerias:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Si quieres rehacer el modelo:

```powershell
python download_datasets.py
python train_multiposture_model.py
```

## 11. Archivos importantes

- `run.py`: abre la aplicacion.
- `infer.py`: contiene la ventana principal y la deteccion en vivo.
- `config.py`: configuracion de tiempos, camara y rutas.
- `train_multiposture_model.py`: entrena el modelo de postura.
- `download_datasets.py`: descarga el dataset.
- `app.py`: muestra el historial con Streamlit.
- `data/`: guarda alertas e hidratacion.
- `models/`: guarda modelos de IA.
- `datasets/`: guarda datasets.
