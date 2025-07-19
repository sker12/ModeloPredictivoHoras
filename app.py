# -*- coding: utf-8 -*-
"""
Script FINAL v4.6 para predecir horas de ejecución.
MODIFICADO: Sube el archivo procesado a un Dataset público para descarga directa.
CORREGIDO: Se añade limpieza automática de nombres de columnas para evitar KeyError.
MODIFICADO PARA DOKPLOY: Configurado para despliegue en producción.
"""

# --- 1. IMPORTACIÓN DE LIBRERÍAS ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import gradio as gr
import warnings
import os
from huggingface_hub import HfApi, hf_hub_download
import tempfile
from pathlib import Path
import datetime

warnings.filterwarnings('ignore')

# --- 2. CARGA Y PREPARACIÓN DE DATOS ---

# Obtener los tokens guardados en los "Secrets" del Space o variables de entorno
hf_token_read = os.getenv("HF_TOKEN")      # Token para lectura de datos
hf_token_write = os.getenv("HF_WRITE_TOKEN") # Token para escritura de resultados

# ⬇️⬇️⬇️ ¡IMPORTANTE! REEMPLAZA ESTOS VALORES POR LOS TUYOS ⬇️⬇️⬇️
# El ID del Space PRIVADO que contiene el archivo de datos
REPO_ID_DATA = "258Yeison258/Modelo-Predictivo-horas"
FILENAME_DATA = "BDgeneral.xlsx"

# ⬇️⬇️⬇️ ¡IMPORTANTE! CONFIGURA TU REPOSITORIO PÚBLICO DE DESTINO ⬇️⬇️⬇️
REPO_ID_RESULTS = "258Yeison258/resultados-modelo-predictivo" # Reemplaza con tu repo de datasets

# Verificar que los tokens estén configurados
if not hf_token_read:
    print("⚠️  ADVERTENCIA: No se encontró el HF_TOKEN en las variables de entorno.")
    print("   La aplicación funcionará en modo demo limitado.")
if not hf_token_write:
    print("⚠️  ADVERTENCIA: No se encontró el HF_WRITE_TOKEN en las variables de entorno.")
    print("   La función de descarga de archivos no estará disponible.")

# Inicializar DataFrame vacío por defecto
df = pd.DataFrame()

try:
    if hf_token_read:
        print(f"Intentando descargar {FILENAME_DATA} desde el Space {REPO_ID_DATA}...")
        local_file_path = hf_hub_download(
            repo_id=REPO_ID_DATA,
            filename=FILENAME_DATA,
            repo_type="space",
            token=hf_token_read
        )
        df = pd.read_excel(local_file_path)
        print("✅ Datos cargados exitosamente desde el Space privado.")

        # --- NUEVO BLOQUE DE LIMPIEZA DE COLUMNAS ---
        print("Columnas originales:", df.columns.tolist())
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=False).str.replace('.', '_', regex=False)
        print("Columnas limpiadas:", df.columns.tolist())
        # ---------------------------------------------

        df.dropna(inplace=True)
    else:
        print("⚠️  Ejecutando en modo demo sin datos reales.")
        # Crear datos de ejemplo para demostración
        df = pd.DataFrame({
            'etapa': ['Diseño', 'Construcción', 'Acabados'] * 10,
            'disciplina': ['Arquitectura', 'Estructural', 'MEP'] * 10,
            'actividades': ['Planos', 'Cálculos', 'Instalación'] * 10,
            'area_proyecto_m2': np.random.uniform(100, 1000, 30),
            'valor_del_contrato': np.random.uniform(50000, 500000, 30),
            'valor_por_m2': np.random.uniform(500, 2000, 30),
            'codigo': [f'DEMO-{i:03d}' for i in range(30)],
            'horas_ejecutadas': np.random.uniform(50, 500, 30),
            'horas_planeadas': np.random.uniform(40, 450, 30)
        })

except Exception as e:
    print(f"❌ Error al cargar o limpiar el archivo de datos: {e}")
    # Crear datos de ejemplo para que la app funcione
    df = pd.DataFrame({
        'etapa': ['Diseño', 'Construcción', 'Acabados'] * 10,
        'disciplina': ['Arquitectura', 'Estructural', 'MEP'] * 10,
        'actividades': ['Planos', 'Cálculos', 'Instalación'] * 10,
        'area_proyecto_m2': np.random.uniform(100, 1000, 30),
        'valor_del_contrato': np.random.uniform(50000, 500000, 30),
        'valor_por_m2': np.random.uniform(500, 2000, 30),
        'codigo': [f'ERROR-{i:03d}' for i in range(30)],
        'horas_ejecutadas': np.random.uniform(50, 500, 30),
        'horas_planeadas': np.random.uniform(40, 450, 30)
    })


# --- 3. DEFINICIÓN DE CARACTERÍSTICAS Y PREPROCESAMIENTO (CON NOMBRES LIMPIOS) ---
if not df.empty:
    target = 'horas_ejecutadas' # <--- CAMBIO A MINÚSCULAS
    y = df[target]
    # <--- CAMBIO A MINÚSCULAS ---
    X = df.drop(columns=['codigo', 'horas_ejecutadas', 'horas_planeadas'])

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. BÚSQUEDA DE HIPERPARÁMETROS Y SELECCIÓN DE MODELO ---
    print("Iniciando optimización de modelos...")
    # (El código de entrenamiento no necesita cambios, ya que opera sobre las variables X e y)
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(random_state=42))])
    param_grid_rf = {'regressor__n_estimators': [100, 200], 'regressor__max_depth': [None, 10, 20], 'regressor__min_samples_leaf': [1, 2, 4], 'regressor__max_features': ['sqrt', 'log2']}
    grid_search_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)

    gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(random_state=42))])
    param_grid_gb = {'regressor__n_estimators': [100, 200], 'regressor__learning_rate': [0.05, 0.1], 'regressor__max_depth': [3, 5, 8], 'regressor__subsample': [0.8, 1.0]}
    grid_search_gb = GridSearchCV(gb_pipeline, param_grid_gb, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search_gb.fit(X_train, y_train)

    if -grid_search_rf.best_score_ < -grid_search_gb.best_score_:
        best_model_pipeline = grid_search_rf.best_estimator_
        model_name = "Random Forest (Optimizado)"
    else:
        best_model_pipeline = grid_search_gb.best_estimator_
        model_name = "Gradient Boosting (Optimizado)"

    print(f"🏆 Modelo seleccionado: {model_name}")
    y_pred = best_model_pipeline.predict(X_test)
    final_mae_test = mean_absolute_error(y_test, y_pred)
    print(f"Error Absoluto Medio (en prueba): {final_mae_test:.2f} horas")
    print("-" * 30)

    # --- 5. CREAR CONJUNTOS DE VALORES VÁLIDOS (CON NOMBRES LIMPIOS) ---
    etapas_validas = set(df['etapa'].unique())
    disciplinas_validas = set(df['disciplina'].unique())
    actividades_validas = set(df['actividades'].unique())

else: # Si el DataFrame está vacío
    X = pd.DataFrame() # Para que X.columns no falle
    model_name = "N/A (Error de carga)"
    final_mae_test = 0
    etapas_validas = set()
    disciplinas_validas = set()
    actividades_validas = set()


print(f"📊 Valores válidos cargados:")
print(f"  - Etapas: {len(etapas_validas)} valores únicos")
print(f"  - Disciplinas: {len(disciplinas_validas)} valores únicos")
print(f"  - Actividades: {len(actividades_validas)} valores únicos")

# --- 6. FUNCIÓN DE VALIDACIÓN ---
def validate_input_data(etapa, disciplina, actividad):
    if df.empty:
        return False, ["Error: No se pudieron cargar los datos de validación."]
    errors = []
    if etapa not in etapas_validas: errors.append(f"ETAPA '{etapa}' no existe")
    if disciplina not in disciplinas_validas: errors.append(f"DISCIPLINA '{disciplina}' no existe")
    if actividad not in actividades_validas: errors.append(f"ACTIVIDAD '{actividad}' no existe")
    return len(errors) == 0, errors

# --- 7. FUNCIONES PARA LA INTERFAZ ---
def predict_hours(etapa, disciplina, actividad, area_proyecto, valor_contrato, valor_m2):
    is_valid, errors = validate_input_data(etapa, disciplina, actividad)
    if not is_valid:
        print(f"❌ Datos inválidos detectados: {'; '.join(errors)}")
        return -1
    area_proyecto = area_proyecto or 0
    valor_contrato = valor_contrato or 0
    valor_m2 = valor_m2 or 0
    try:
        input_data = pd.DataFrame([[etapa, disciplina, actividad, area_proyecto, valor_contrato, valor_m2]], columns=X.columns)
        prediction = best_model_pipeline.predict(input_data)
        return round(prediction[0], 2)
    except Exception as e:
        print(f"❌ Error en la predicción: {e}")
        return -1

def update_valor_m2(area, valor_contrato):
    area = area or 0
    valor_contrato = valor_contrato or 0
    if area > 0:
        return round(valor_contrato / area, 2)
    return 0

def process_excel_file(file_obj, default_area, default_valor):
    if file_obj is None:
        return "Por favor, sube un archivo Excel primero.", gr.update(visible=False)
    
    # Verificar si tenemos token de escritura
    if not hf_token_write:
        return "❌ Error: Token de escritura no configurado. Contacta al administrador.", gr.update(visible=False)
    
    try:
        df_input = pd.read_excel(file_obj.name)
        
        # --- APLICAR LA MISMA LIMPIEZA AL ARCHIVO DEL USUARIO ---
        df_input.columns = df_input.columns.str.strip().str.lower().str.replace(' ', '_', regex=False).str.replace('.', '_', regex=False)
        
        # Verificar columnas requeridas (en minúsculas)
        required_columns = ['etapa', 'disciplina', 'actividades']
        missing_columns = [col for col in required_columns if col not in df_input.columns]
        if missing_columns:
            return f"Error: Faltan columnas. Se requieren: {missing_columns}", gr.update(visible=False)

        # Usar los nombres de columna limpios para crear las nuevas columnas
        df_input['horas_ejecutadas'] = 0.0
        df_input['estado_validacion'] = ""
        
        filas_con_error = 0
        for index, row in df_input.iterrows():
            # Acceder a las columnas por su nombre limpio
            etapa = str(row.get('etapa', '')).strip()
            disciplina = str(row.get('disciplina', '')).strip()
            actividad = str(row.get('actividades', '')).strip()
            
            is_valid, validation_errors = validate_input_data(etapa, disciplina, actividad)
            
            if not is_valid:
                df_input.loc[index, 'horas_ejecutadas'] = -1
                df_input.loc[index, 'estado_validacion'] = f"ERROR: {'; '.join(validation_errors)}"
                filas_con_error += 1
            else:
                area = default_area or 0
                valor = default_valor or 0
                valor_m2 = valor / area if area > 0 else 0
                horas_estimadas = predict_hours(etapa, disciplina, actividad, area, valor, valor_m2)
                df_input.loc[index, 'horas_ejecutadas'] = horas_estimadas
                df_input.loc[index, 'estado_validacion'] = "OK - Predicción exitosa"

        # Lógica de subida del archivo
        original_filename = Path(file_obj.name).stem
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        updated_filename = f"{original_filename}_procesado_{timestamp}.xlsx"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_f:
            df_input.to_excel(temp_f.name, index=False)
            temp_file_path = temp_f.name

        print(f"Subiendo {updated_filename} a {REPO_ID_RESULTS}...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=temp_file_path,
            path_in_repo=updated_filename,
            repo_id=REPO_ID_RESULTS,
            repo_type="dataset",
            token=hf_token_write,
        )
        print("✅ Archivo subido exitosamente.")
        os.remove(temp_file_path)

        download_url = f"https://huggingface.co/datasets/{REPO_ID_RESULTS}/resolve/main/{updated_filename}"
        
        success_message = f"""✅ PROCESAMIENTO COMPLETADO
- Filas procesadas: {len(df_input)}
- Filas con errores: {filas_con_error}
- Archivo subido como: {updated_filename}"""
        
        download_link_md = f"**[➡️ Haz clic aquí para descargar tu archivo]({download_url})**"
        
        return success_message, gr.update(value=download_link_md, visible=True)

    except Exception as e:
        error_msg = f"❌ Error al procesar el archivo: {str(e)}"
        print(error_msg)
        return error_msg, gr.update(visible=False)

# --- 8. CREACIÓN DE LA INTERFAZ DE USUARIO CON GRADIO ---
etapas_unicas = sorted(list(etapas_validas))
disciplinas_unicas = sorted(list(disciplinas_validas))
actividades_unicas = sorted(list(actividades_validas))

# Configurar status de la aplicación
status_message = "🟢 Aplicación funcionando correctamente"
if not hf_token_read:
    status_message = "🟡 Aplicación en modo demo (sin datos reales)"
if not hf_token_write:
    status_message += " - Descarga de archivos deshabilitada"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 👷 Estimador de Horas de Proyecto v1.4 (Dokploy)")
    gr.Markdown(f"**Estado:** {status_message}")
    gr.Markdown(
        """
        **Opciones disponibles:**
        1. **Predicción Individual:** Introduce los datos manualmente para obtener una estimación
        2. **Procesamiento Masivo:** Sube un archivo Excel para procesar múltiples registros
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("Predicción Individual"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Introduce los Datos")
                    input_etapa = gr.Dropdown(choices=etapas_unicas, label="Etapa del Proyecto")
                    input_disciplina = gr.Dropdown(choices=disciplinas_unicas, label="Disciplina")
                    input_actividad = gr.Dropdown(choices=actividades_unicas, label="Actividad Específica")
                    
                    gr.Markdown("### Datos del Proyecto (El Valor por m² se calcula automáticamente)")
                    input_area = gr.Number(label="Área del Proyecto (m²)", value=0)
                    input_valor = gr.Number(label="Valor del Contrato ($)", value=0)
                    input_valor_m2 = gr.Number(label="Valor por m² (Calculado)", interactive=False, value=0)
                    
                    predict_button = gr.Button("Estimar Horas", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### Resultado")
                    output_hours = gr.Number(label="Horas Estimadas", interactive=False, value=0)
                    
                    gr.Markdown(
                        """
                        **Interpretación de Resultados:**
                        - **Número positivo:** Predicción exitosa (horas estimadas)
                        - **-1:** Error en validación (valor no existe en base de datos)
                        """
                    )

                    gr.Markdown(
                        f"""
                        **Información del Modelo**
                        - **Modelo en uso:** {model_name}
                        - **Error Absoluto Medio:** {final_mae_test:.2f} horas
                        - **Etapas válidas:** {len(etapas_validas)}
                        - **Disciplinas válidas:** {len(disciplinas_validas)}
                        - **Actividades válidas:** {len(actividades_validas)}
                        """
                    )
        
        with gr.TabItem("Procesamiento Masivo"):
            # --- BLOQUE MODIFICADO ---
            gr.Markdown(
                """
                ### Sube tu archivo Excel para procesar múltiples registros

                **🧾 VALIDACIÓN AUTOMÁTICA:**
                - El sistema verificará que cada valor de `ETAPA`, `DISCIPLINA` y `ACTIVIDADES` exista en la base de datos.
                - Si un valor no existe, se asignará `-1` en la columna "Horas_Ejecutadas".
                - Se agregará una columna "Estado_Validacion" con detalles de errores.

                **Requisitos del archivo:**
                - Debe ser un archivo Excel (.xlsx).
                - Debe contener las columnas: `ETAPA`, `DISCIPLINA`, `ACTIVIDADES`.
                - Se agregarán automáticamente las columnas `Horas_Ejecutadas` y `Estado_Validacion`.
                - El archivo procesado se subirá al repositorio público para su descarga.
                
                ---
                ⚠️ **Nota:** El archivo se guardará temporalmente para su descarga.
                """
            )
            # --- FIN DEL BLOQUE MODIFICADO ---
            with gr.Row():
                with gr.Column(scale=2):
                    excel_file = gr.File(label="Seleccionar archivo Excel", file_types=[".xlsx"])
                    batch_area = gr.Number(label="Área del Proyecto por Defecto (m²)", value=0)
                    batch_valor = gr.Number(label="Valor del Contrato por Defecto ($)", value=0)
                    process_button = gr.Button("🚀 Procesar Archivo Excel", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Estado del Procesamiento")
                    process_status = gr.Textbox(label="Estado", interactive=False, lines=5)
                    download_link = gr.Markdown(visible=False)

    # --- EVENTOS DE LA INTERFAZ ---
    input_area.change(fn=update_valor_m2, inputs=[input_area, input_valor], outputs=input_valor_m2)
    input_valor.change(fn=update_valor_m2, inputs=[input_area, input_valor], outputs=input_valor_m2)
    predict_button.click(fn=predict_hours, inputs=[input_etapa, input_disciplina, input_actividad, input_area, input_valor, input_valor_m2], outputs=output_hours)
    
    process_button.click(
        fn=process_excel_file,
        inputs=[excel_file, batch_area, batch_valor],
        outputs=[process_status, download_link]
    )

# --- 9. CONFIGURACIÓN PARA PRODUCCIÓN EN DOKPLOY ---
if __name__ == "__main__":
    # Configuración específica para producción
    server_name = os.getenv("GRADIO_SERVER_NAME", "paginaweb-modelopredictivo-dlyi5o-c14057-31-59-40-250.traefik.me")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    print(f"🚀 Iniciando aplicación en {server_name}:{server_port}")
    print(f"📊 Modo: {'Producción' if hf_token_read else 'Demo'}")
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=True,
        show_error=True,
        quiet=False,
        favicon_path=None,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_keyfile_password=None
    )
                
