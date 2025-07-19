#!/usr/bin/env python3
"""
Health Check Script para la aplicación de predicción de horas
Verifica que la aplicación esté funcionando correctamente
"""

import requests
import sys
import os
import time
import json
from datetime import datetime

def health_check(host="localhost", port=7860, timeout=30):
    """
    Realiza un health check de la aplicación Gradio
    """
    url = f"http://{host}:{port}"
    
    print(f"🔍 Verificando salud de la aplicación en {url}")
    print(f"⏰ Timestamp: {datetime.now().isoformat()}")
    print("-" * 50)
    
    try:
        # Test 1: Verificar que el servidor responde
        print("1. Verificando conectividad del servidor...")
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            print("   ✅ Servidor respondiendo correctamente")
        else:
            print(f"   ❌ Servidor respondió con código: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ No se puede conectar al servidor")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout después de {timeout} segundos")
        return False
    except Exception as e:
        print(f"   ❌ Error inesperado: {e}")
        return False
    
    try:
        # Test 2: Verificar la API de Gradio
        print("2. Verificando API de Gradio...")
        api_url = f"{url}/api/"
        api_response = requests.get(api_url, timeout=timeout)
        
        if api_response.status_code == 200:
            print("   ✅ API de Gradio funcionando")
        else:
            print(f"   ⚠️ API respondió con código: {api_response.status_code}")
            
    except Exception as e:
        print(f"   ⚠️ Error al verificar API: {e}")
    
    # Test 3: Verificar variables de entorno críticas
    print("3. Verificando configuración...")
    
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "HF_WRITE_TOKEN": os.getenv("HF_WRITE_TOKEN"),
        "GRADIO_SERVER_NAME": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        "GRADIO_SERVER_PORT": os.getenv("GRADIO_SERVER_PORT", "7860")
    }
    
    for var_name, var_value in env_vars.items():
        if var_value:
            if "TOKEN" in var_name:
                print(f"   ✅ {var_name}: Configurado (oculto por seguridad)")
            else:
                print(f"   ✅ {var_name}: {var_value}")
        else:
            if "TOKEN" in var_name:
                print(f"   ⚠️ {var_name}: No configurado (funcionalidad limitada)")
            else:
                print(f"   ❌ {var_name}: No configurado")
    
    # Test 4: Verificar recursos del sistema
    print("4. Verificando recursos del sistema...")
    
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"   📊 CPU: {cpu_percent}%")
        print(f"   💾 RAM: {memory.percent}% usado ({memory.used // (1024**3)}GB/{memory.total // (1024**3)}GB)")
        print(f"   💿 Disco: {disk.percent}% usado ({disk.used // (1024**3)}GB/{disk.total // (1024**3)}GB)")
        
        # Alertas de recursos
        if cpu_percent > 80:
            print("   ⚠️ Alto uso de CPU")
        if memory.percent > 80:
            print("   ⚠️ Alta utilización de memoria")
        if disk.percent > 90:
            print("   ⚠️ Espacio en disco bajo")
            
    except ImportError:
        print("   ⚠️ psutil no disponible, no se pueden verificar recursos")
    except Exception as e:
        print(f"   ⚠️ Error al verificar recursos: {e}")
    
    print("-" * 50)
    print("✅ Health check completado exitosamente")
    return True

def create_health_endpoint():
    """
    Crea un endpoint simple de health check que se puede agregar a app.py
    """
    health_endpoint_code = '''
# Agregar este código a tu app.py para crear un endpoint de health check

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.datetime.now().isoformat(),
                "service": "prediccion-horas",
                "version": "1.4"
            }
            
            self.wfile.write(json.dumps(health_status).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Silenciar logs del health check
        pass

def start_health_server():
    """Inicia el servidor de health check en un hilo separado"""
    server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
    server.serve_forever()

# Agregar al final de tu app.py, antes de demo.launch():
health_thread = threading.Thread(target=start_health_server, daemon=True)
health_thread.start()
print("🏥 Health check endpoint iniciado en puerto 8080")
'''
    
    return health_endpoint_code

def main():
    """Función principal del health check"""
    
    # Argumentos de línea de comandos
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 7860
    
    # Si se pasa el argumento "wait", esperar a que el servicio esté disponible
    if len(sys.argv) > 3 and sys.argv[3] == "wait":
        print(f"⏳ Esperando a que el servicio esté disponible en {host}:{port}...")
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            try:
                response = requests.get(f"http://{host}:{port}", timeout=5)
                if response.status_code == 200:
                    print("✅ Servicio disponible!")
                    break
            except:
                pass
            
            attempt += 1
            time.sleep(10)
            print(f"   Intento {attempt}/{max_attempts}...")
        
        if attempt >= max_attempts:
            print("❌ Timeout esperando al servicio")
            sys.exit(1)
    
    # Ejecutar health check
    success = health_check(host, port)
    
    # Salir con código apropiado
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
