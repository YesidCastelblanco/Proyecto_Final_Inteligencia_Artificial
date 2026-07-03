"""
Sistema de auditoría completo para modelos
Registra: datos de entrada, preprocesamiento, predicciones, posibles problemas
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
import traceback

def setup_model_logger(model_name):
    """Configura logger específico para cada modelo"""
    logger = logging.getLogger(f"model_audit_{model_name}")
    logger.setLevel(logging.DEBUG)
    
    # Evitar duplicar handlers
    if not logger.handlers:
        # Handler para archivo
        fh = logging.FileHandler(f"model_audit_{model_name}_{datetime.now().strftime('%Y%m%d')}.log")
        fh.setLevel(logging.DEBUG)
        
        # Handler para consola
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

def audit_model_execution(model_func):
    """Decorator para auditar ejecución de modelos"""
    def wrapper(df_train, df_future, *args, **kwargs):
        model_name = model_func.__name__
        logger = setup_model_logger(model_name)
        
        # === AUDITORÍA INICIAL ===
        logger.info("🚀 === INICIO EJECUCIÓN MODELO ===")
        logger.info(f"📊 Modelo: {model_name}")
        logger.info(f"📥 df_train shape: {df_train.shape}")
        logger.info(f"📤 df_future shape: {df_future.shape}")
        
        # Auditoría de datos de entrada
        _audit_data_quality(logger, df_train, "TRAIN")
        _audit_data_quality(logger, df_future, "FUTURE")
        
        try:
            # Ejecutar modelo
            start_time = datetime.now()
            result = model_func(df_train, df_future, *args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # === AUDITORÍA DE SALIDA ===
            logger.info(f"⏱️  Tiempo ejecución: {execution_time:.2f}s")
            _audit_prediction_result(logger, result, model_name)
            
            return result
            
        except Exception as e:
            # === AUDITORÍA DE ERRORES ===
            logger.error(f"💥 ERROR en modelo: {str(e)}")
            logger.error(f"🔍 Stack trace: {traceback.format_exc()}")
            raise
    
    return wrapper

def _audit_data_quality(logger, df, dataset_type):
    """Audita calidad de datos"""
    logger.info(f"🔍 AUDITORÍA DATOS {dataset_type}:")
    logger.info(f"   📏 Shape: {df.shape}")
    logger.info(f"   🔢 Columnas: {list(df.columns)}")
    
    # Estadísticas básicas
    if not df.empty:
        logger.info(f"   📈 Rango fechas: {df.index.min()} a {df.index.max()}")
        logger.info(f"   🎯 Target (Close): min={df['ETH-USD_Close'].min():.2f}, max={df['ETH-USD_Close'].max():.2f}")
        
        # Verificar NaN
        nan_counts = df.isna().sum()
        total_nan = nan_counts.sum()
        if total_nan > 0:
            logger.warning(f"   ⚠️  NaN encontrados: {total_nan} total")
            high_nan_cols = nan_counts[nan_counts > 0]
            for col, count in high_nan_cols.items():
                logger.warning(f"      - {col}: {count} NaN ({count/len(df)*100:.1f}%)")
        
        # Verificar infinitos
        inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
        inf_counts = inf_mask.sum().sum()
        if inf_counts > 0:
            logger.error(f"   ❗ INFINITOS encontrados: {inf_counts}")
        
        # Verificar valores extremos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Primeras 5 columnas numéricas
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                if outliers > 0:
                    logger.info(f"   📊 {col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")

def _audit_prediction_result(logger, result, model_name):
    """Audita el resultado de la predicción"""
    logger.info(f"📤 AUDITORÍA PREDICCIÓN {model_name}:")
    
    if result is None:
        logger.error("   ❌ Predicción: None")
        return
    
    if isinstance(result, (int, float)):
        logger.info(f"   ✅ Predicción: {result:.6f}")
        # Verificar si es un valor razonable
        if abs(result) > 1e6:  # Valor extremadamente alto
            logger.warning("   ⚠️  Predicción: Valor potencialmente irreal")
        elif np.isnan(result):
            logger.error("   ❌ Predicción: NaN")
        elif np.isinf(result):
            logger.error("   ❌ Predicción: Infinito")
            
    elif isinstance(result, (list, np.ndarray, pd.Series)):
        logger.info(f"   📦 Predicción: Array con {len(result)} elementos")
        logger.info(f"   📊 Stats: min={np.min(result):.6f}, max={np.max(result):.6f}, mean={np.mean(result):.6f}")
        
        # Verificar elementos
        nan_count = np.sum(pd.isna(result))
        if nan_count > 0:
            logger.error(f"   ❌ Array contiene {nan_count} NaN")
            
    elif isinstance(result, pd.DataFrame):
        logger.info(f"   🗂️  Predicción: DataFrame {result.shape}")
        logger.info(f"   📋 Columnas: {list(result.columns)}")





En model2_lstm_transformer.py (y los otros modelos):

from src.model_audit_logger import audit_model_execution

@audit_model_execution
def predict_model2(df_train, df_future):
    # Tu código actual permanece igual
    target = "ETH-USD_Close"
    df_train_clean = safe_features(df_train.copy())
    df_future_clean = safe_features(df_future.copy())
    
    # ... resto del código igual


3. También audita el rolling forecast:

En rolling_forecast.py:

from src.model_audit_logger import setup_model_logger

def rolling_expanding_forecast(...):
    # Al inicio de la función
    logger = setup_model_logger("rolling_forecast")
    logger.info("🎯 INICIANDO ROLLING FORECAST")
    logger.info(f"Modelo: {modelo_id}, Horizonte: {horizonte}")
    
    # En cada iteración del loop
    while fecha_bt_actual <= fecha_bt_final:
        logger.info(f"🔄 Iteración {consecutivo} - Corte: {fecha_bt_actual.date()}")
        
        # ... código existente
        
        # Después de cada predicción
        logger.info(f"✅ Predicción {consecutivo}: close={close_pred:.2f}, high={high_pred:.2f}, low={low_pred:.2f}")



        4. Qué información obtendrás:

        ✅ DATOS DE ENTRADA: shapes, columnas, rangos de fechas
✅ CALIDAD: NaN, infinitos, outliers por columna  
✅ PREPROCESAMIENTO: Cómo transforma los datos
✅ PERFORMANCE: Tiempos de ejecución
✅ PREDICCIÓN: Valores, distribuciones, posibles problemas
✅ ERRORES: Stack traces completos cuando falle


5. Para análisis posterior:

# Analizar logs generados
def analyze_model_logs():
    with open("model_audit_predict_model2_20241127.log", "r") as f:
        logs = f.readlines()
    
    # Buscar patrones problemáticos
    problems = [line for line in logs if "NaN" in line or "ERROR" in line or "WARNING" in line]
    return problems