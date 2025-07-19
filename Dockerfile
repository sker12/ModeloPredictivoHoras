FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade gradio
# Copiar el código de la aplicación
COPY . .

# Crear directorio para archivos temporales
RUN mkdir -p /tmp/gradio

# Exponer el puerto que usa Gradio
EXPOSE 7860

# Configurar variables de entorno para Gradio
ENV GRADIO_SERVER_NAME=paginaweb-modelopredictivo-dlyi5o-c14057-31-59-40-250.traefik.me
ENV GRADIO_SERVER_PORT=7860

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
