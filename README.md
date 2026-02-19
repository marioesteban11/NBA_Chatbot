NBA Chatbot

Este repositorio contiene el código, los datos y los recursos necesarios para la ejecución del Trabajo Fin de Máster basado en la construcción de un chatbot especializado en información de la NBA.
A continuación se detallan los pasos necesarios para reproducir el proyecto de forma correcta.


1. Descarga del repositorio
Clone o descargue este repositorio en su equipo local:
git clone https://github.com/marioesteban11/NBA_Chatbot


También puede utilizar la opción Download ZIP disponible en GitHub.

2. Generación del dataset procesado
Antes de ejecutar la aplicación, es necesario generar el archivo nba_cleaned_final, que será utilizado por el chatbot.
Para ello:
- Acceda a la carpeta del proyecto.
- Abra el archivo Jupyter Notebook llamado Mario.ipynb.
- Ejecútelo completamente (todas las celdas).
- El notebook generará automáticamente el archivo procesado requerido para el funcionamiento del chatbot.

3. Ejecución del chatbot
Una vez generado el archivo de datos, abra una terminal en la carpeta raíz del proyecto y ejecute:
streamlit run app.py


Esto iniciará la aplicación en Streamlit y permitirá interactuar con el chatbot desde el navegador.


Notas adicionales
- Es recomendable utilizar un entorno virtual (Conda o venv) para instalar las dependencias del proyecto.
- En caso de cualquier incidencia, revisar las versiones de las librerías indicadas en el archivo requirements.txt.
