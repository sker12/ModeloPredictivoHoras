# Usa una imagen base de Python. Siempre es buena práctica usar una versión específica.
# Puedes elegir una versión más ligera como 'slim-buster' o 'alpine' si el tamaño es crítico.
FROM python:3.9-slim-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de requerimientos primero para aprovechar la caché de Docker
# Esto instala las dependencias antes de copiar todo el código de la aplicación
COPY requirements.txt .

# Instala las dependencias de Python
# La opción --no-cache-dir reduce el tamaño de la imagen final
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu código fuente al directorio de trabajo
COPY . .

# Expone el puerto que usa Gradio (por defecto es 7860)
EXPOSE 7860

# Define las variables de entorno para los tokens de Hugging Face.
# Es MEJOR que Dokploy gestione estas como "Secrets" o "Environment Variables"
# y no las pongas directamente aquí para seguridad.
# Si las pones aquí, serán visibles en la imagen Docker.
# ENV HF_TOKEN="tu_token_de_lectura_aqui"
# ENV HF_WRITE_TOKEN="tu_token_de_escritura_aqui"

# Comando para ejecutar la aplicación cuando el contenedor se inicie
# CMD ["python", "app.py"] # Este es el formato de "exec" preferido
ENTRYPOINT ["python", "app.py"] # ENTRYPOINT es similar a CMD, pero asegura que el comando se ejecuta y no se sobrescribe fácilmente.