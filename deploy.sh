#!/bin/bash

# Script de despliegue automatizado para Dokploy
# Autor: Assistant
# Versión: 1.0

echo "🚀 Script de Despliegue Automatizado para Dokploy"
echo "================================================="

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar mensajes coloridos
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar si estamos en el directorio correcto
if [ ! -f "app.py" ]; then
    print_error "No se encontró app.py en el directorio actual."
    print_error "Asegúrate de estar en el directorio raíz del proyecto."
    exit 1
fi

print_status "Verificando archivos del proyecto..."

# Crear Dockerfile si no existe
if [ ! -f "Dockerfile" ]; then
    print_warning "Dockerfile no encontrado. Creando uno básico..."
    cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear directorio para archivos temporales
RUN mkdir -p /tmp/gradio

# Exponer el puerto que usa Gradio
EXPOSE 7860

# Configurar variables de entorno para Gradio
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
EOF
    print_success "Dockerfile creado exitosamente"
fi

# Crear .dockerignore si no existe
if [ ! -f ".dockerignore" ]; then
    print_warning ".dockerignore no encontrado. Creando uno básico..."
    cat > .dockerignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment variables
.env
.env.local
.env.*.local

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Git
.git/
.gitignore

# Documentation
README.md
*.md

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp
EOF
    print_success ".dockerignore creado exitosamente"
fi

# Verificar requirements.txt
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt no encontrado."
    print_error "Crea un archivo requirements.txt con las dependencias necesarias."
    exit 1
fi

print_success "Verificación de archivos completada"

# Verificar si Git está inicializado
if [ ! -d ".git" ]; then
    print_warning "Repositorio Git no inicializado. Inicializando..."
    git init
    git add .
    git commit -m "Initial commit for Dokploy deployment"
    print_success "Repositorio Git inicializado"
fi

# Mostrar checklist de despliegue
echo ""
echo "📋 CHECKLIST DE DESPLIEGUE"
echo "=========================="
echo ""
echo "Antes de proceder con Dokploy, verifica que tengas:"
echo ""
echo "✅ Archivos del proyecto:"
echo "   - app.py (tu aplicación principal)"
echo "   - requirements.txt (dependencias Python)"
echo "   - Dockerfile (creado automáticamente)"
echo "   - .dockerignore (creado automáticamente)"
echo ""
echo "🔑 Variables de entorno necesarias:"
echo "   - HF_TOKEN: Token de lectura de Hugging Face"
echo "   - HF_WRITE_TOKEN: Token de escritura de Hugging Face"
echo ""
echo "🌐 Configuración de servidor:"
echo "   - VPS con Docker instalado"
echo "   - Dokploy instalado y funcionando"
echo "   - Puerto 7860 disponible"
echo ""
echo "📚 Repositorio Git:"
echo "   - Código subido a GitHub/GitLab"
echo "   - Repositorio accesible públicamente o con credenciales"
echo ""

# Preguntar si quiere continuar con instrucciones de Dokploy
echo ""
read -p "¿Quieres ver las instrucciones paso a paso para Dokploy? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔧 PASOS PARA DOKPLOY"
    echo "===================="
    echo ""
    echo "1. Accede a tu panel de Dokploy: http://tu-servidor-ip:3000"
    echo ""
    echo "2. Crear nueva aplicación:"
    echo "   - Click en 'New Project'"
    echo "   - Selecciona 'Application'"
    echo "   - Elige 'Git' como fuente"
    echo ""
    echo "3. Configurar repositorio:"
    echo "   - Repository URL: $(git remote get-url origin 2>/dev/null || echo 'TU_REPO_URL')"
    echo "   - Branch: main"
    echo "   - Build Path: /"
    echo ""
    echo "4. Configuración de build:"
    echo "   - Dockerfile Path: ./Dockerfile"
    echo "   - Port: 7860"
    echo ""
    echo "5. Variables de entorno (IMPORTANTE):"
    echo "   - HF_TOKEN=tu_token_de_lectura"
    echo "   - HF_WRITE_TOKEN=tu_token_de_escritura"
    echo ""
    echo "6. Desplegar:"
    echo "   - Click en 'Deploy'"
    echo "   - Esperar a que termine el build"
    echo "   - Tu app estará disponible en el dominio configurado"
    echo ""
fi

# Mostrar información del sistema si estamos en el servidor
if command -v docker &> /dev/null; then
    echo ""
    echo "🖥️  INFORMACIÓN DEL SISTEMA"
    echo "=========================="
    echo "Docker versión: $(docker --version)"
    echo "Espacio disponible: $(df -h / | awk 'NR==2{print $4}')"
    echo "Memoria disponible: $(free -h | awk 'NR==2{print $7}')"
fi

echo ""
print_success "Script completado. Tu proyecto está listo para Dokploy!"
echo ""
echo "🔗 Próximos pasos:"
echo "1. Sube tu código a Git si no lo has hecho"
echo "2. Sigue las instrucciones de Dokploy mostradas arriba"
echo "3. Configura las variables de entorno en Dokploy"
echo "4. ¡Despliega tu aplicación!"
echo ""
echo "📞 Si tienes problemas:"
echo "- Revisa los logs en Dokploy > Tu Proyecto > Logs"
echo "- Verifica que las variables de entorno estén configuradas"
echo "- Asegúrate de que tu VPS tenga suficiente memoria (mín 2GB)"
echo ""
