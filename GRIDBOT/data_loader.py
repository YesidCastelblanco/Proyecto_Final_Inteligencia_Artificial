# src/data_loader.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
from utils import log_gridbot_event

def verificar_integridad_fechas(df):
    """Verifica la integridad de las fechas en el DataFrame"""
    print("\n=== VERIFICACIÓN DE FECHAS ===")
    print(f"Primera fecha: {df['fecha'].min()}")
    print(f"Última fecha: {df['fecha'].max()}")
    print(f"Total de registros: {len(df)}")
    
    # Verificar fechas futuras
    fecha_actual = pd.Timestamp.now(tz='UTC')
    fechas_futuras = df[df['fecha'] > fecha_actual]
    if len(fechas_futuras) > 0:
        print(f"⚠️ ADVERTENCIA: {len(fechas_futuras)} registros con fechas futuras")
        print(f"Fechas futuras: {fechas_futuras['fecha'].head(3).tolist()}")
    
    # Verificar duplicados
    duplicados = df[df.duplicated(subset=['fecha'], keep=False)]
    if len(duplicados) > 0:
        print(f"⚠️ ADVERTENCIA: {len(duplicados)} registros duplicados")
    
    return df

def cargar_datos_yfinance(
    simbolo="ETH-USD",
    fecha_inicio=datetime(2018, 1, 1),
    fecha_fin=datetime(2024, 12, 31, 23, 59, 59),
    intervalo="1m",
    cache_dir="cache",
    cache_filename="ETHUSD_1m.parquet",
    log_dir="logs",
    log_filename="logs_gridbot.txt",
    force_refresh=False,
    integrity_check=True,
    expected_interval_seconds=60
):
    print(f"\n=== CARGANDO DATOS DESDE YAHOO FINANCE (CACHE + LOGS ORGANIZADOS) ===")
    log_gridbot_event("Inicio de carga incremental Yahoo Finance.", log_dir, log_filename)

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_filename)

    # Convertir fechas de entrada a datetime con timezone UTC para consistencia
    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)
    
    # Si no tienen timezone, asignar UTC
    if fecha_inicio.tz is None:
        fecha_inicio = fecha_inicio.tz_localize('UTC')
    if fecha_fin.tz is None:
        fecha_fin = fecha_fin.tz_localize('UTC')

    if os.path.exists(cache_path) and not force_refresh:
        df_cache = pd.read_parquet(cache_path)
        df_cache["fecha"] = pd.to_datetime(df_cache["fecha"], errors="coerce")
        
        # Asegurar que las fechas del cache tengan timezone UTC
        if df_cache["fecha"].dt.tz is None:
            df_cache["fecha"] = df_cache["fecha"].dt.tz_localize('UTC')
        
        df_cache = df_cache.dropna(subset=["fecha", "close"])
        min_cache, max_cache = df_cache["fecha"].min(), df_cache["fecha"].max()
        print(f"Cache local detectado: {cache_path}")
        print(f"Cache cubre rango: {min_cache} → {max_cache}")
        log_gridbot_event(f"Cache existente leído: {min_cache} → {max_cache}", log_dir, log_filename)

        if min_cache <= fecha_inicio and max_cache >= fecha_fin:
            print("Cache ya cubre todo el rango solicitado.")
            df_final = df_cache[(df_cache["fecha"] >= fecha_inicio) & (df_cache["fecha"] <= fecha_fin)]
            print(f"Datos finales cargados: {len(df_final)} registros en rango solicitado.\n")
            return df_final
    else:
        df_cache = pd.DataFrame(columns=["fecha", "close"])

    print(f"⬇ Descargando datos desde Yahoo Finance ({simbolo})...")
    log_gridbot_event(f"Iniciando descarga desde Yahoo Finance: {simbolo}", log_dir, log_filename)
    
    try:
        # Convertir fechas a string para yfinance
        start_str = fecha_inicio.strftime("%Y-%m-%d")
        end_str = fecha_fin.strftime("%Y-%m-%d")
        
        print(f"Descargando {simbolo} desde {start_str} hasta {end_str} con intervalo {intervalo}")
        
        ticker = yf.Ticker(simbolo)
        df_new = ticker.history(start=start_str, end=end_str, interval=intervalo, auto_adjust=True)

        if df_new.empty:
            print("⚠️ No se encontraron datos con el rango especificado, intentando descarga máxima...")
            df_new = ticker.history(period="max", interval=intervalo, auto_adjust=True)
            
        if df_new.empty:
            raise ValueError("No se pudieron descargar datos de Yahoo Finance")

        print(f"Datos brutos descargados: {len(df_new)} registros")
        print(f"Columnas disponibles: {list(df_new.columns)}")
        
        df_new = df_new.reset_index()
        print(f"Columnas después de reset_index: {list(df_new.columns)}")

        # DETECCIÓN AUTOMÁTICA DE NOMBRE DE COLUMNA DE FECHA
        fecha_column = None
        for col in df_new.columns:
            col_lower = col.lower()
            if col_lower in ['datetime', 'date', 'index']:
                fecha_column = col
                break
        
        if fecha_column is None:
            # Si no encontramos una columna obvia, usar la primera columna que parezca datetime
            for col in df_new.columns:
                if pd.api.types.is_datetime64_any_dtype(df_new[col]):
                    fecha_column = col
                    break
        
        if fecha_column is None:
            # Último recurso: usar la primera columna
            fecha_column = df_new.columns[0]
            print(f"⚠️ No se detectó columna de fecha claramente, usando: {fecha_column}")

        print(f"Usando columna de fecha: {fecha_column}")

        # Renombrar columnas
        df_new.rename(columns={fecha_column: 'fecha', 'Close': 'close'}, inplace=True)
        
        # Asegurar que tenemos las columnas necesarias
        if 'close' not in df_new.columns:
            # Buscar columna de precio de cierre
            for col in df_new.columns:
                if 'close' in col.lower():
                    df_new.rename(columns={col: 'close'}, inplace=True)
                    break
        
        # Mantener solo las columnas necesarias
        required_columns = ['fecha', 'close']
        available_columns = [col for col in required_columns if col in df_new.columns]
        
        if len(available_columns) < 2:
            raise ValueError(f"No se encontraron las columnas requeridas. Disponibles: {list(df_new.columns)}")
        
        df_new = df_new[available_columns].copy()
        
        # FECHAS - MANEJO DE TIMEZONE
        try:
            # Intentar parsing directo
            df_new['fecha'] = pd.to_datetime(df_new['fecha'], errors='coerce')
            
            # Si hay muchos NaT, intentar formato específico
            if df_new['fecha'].isna().sum() > len(df_new) * 0.1:  # Si más del 10% son NaT
                print("⚠️ Problemas con parsing de fechas, intentando formato específico...")
                # Intentar diferentes formatos comunes de Yahoo Finance
                for fmt in ['%Y-%m-%d %H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                    try:
                        df_new['fecha'] = pd.to_datetime(df_new['fecha'], format=fmt, errors='coerce')
                        if df_new['fecha'].isna().sum() == 0:
                            print(f"✅ Formato de fecha detectado: {fmt}")
                            break
                    except:
                        continue
            
            # Eliminar filas con fechas inválidas
            before_clean = len(df_new)
            df_new = df_new.dropna(subset=['fecha'])
            after_clean = len(df_new)
            if after_clean < before_clean:
                print(f"⚠️ Eliminadas {before_clean - after_clean} filas con fechas inválidas")
                
            # Si las fechas no tienen timezone, asignar UTC
            if df_new['fecha'].dt.tz is None:
                df_new['fecha'] = df_new['fecha'].dt.tz_localize('UTC')
            else:
                # Si ya tienen timezone, convertir a UTC
                df_new['fecha'] = df_new['fecha'].dt.tz_convert('UTC')
                
        except Exception as e:
            print(f"❌ Error procesando fechas: {e}")
            raise
        
        df_new = df_new.dropna().reset_index(drop=True)

        print(f"Nuevos registros descargados: {len(df_new)}")
        print(f"Rango de fechas descargado: {df_new['fecha'].min()} a {df_new['fecha'].max()}")
        print(f"Timezone de fechas: {df_new['fecha'].dt.tz}")
        log_gridbot_event(f"Nuevos registros descargados: {len(df_new)}", log_dir, log_filename)

        if df_cache.empty:
            df_cache = df_new.copy()
        else:
            # Asegurar que el cache también tenga timezone UTC
            if df_cache["fecha"].dt.tz is None:
                df_cache["fecha"] = df_cache["fecha"].dt.tz_localize('UTC')
            df_cache = pd.concat([df_cache, df_new], ignore_index=True)

    except Exception as e:
        print(f"❌ Error descargando datos de Yahoo Finance: {e}")
        log_gridbot_event(f"Error en descarga Yahoo Finance: {e}", log_dir, log_filename)
        
        # Mostrar información de debugging más detallada
        try:
            ticker = yf.Ticker(simbolo)
            df_debug = ticker.history(period="1mo", interval=intervalo, auto_adjust=True)
            print(f"Columnas disponibles en Yahoo Finance: {list(df_debug.columns)}")
            if not df_debug.empty:
                df_debug_reset = df_debug.reset_index()
                print(f"Columnas después de reset_index: {list(df_debug_reset.columns)}")
                print(f"Tipos de datos: {df_debug_reset.dtypes}")
        except Exception as debug_e:
            print(f"Error en debugging: {debug_e}")
            
        if os.path.exists(cache_path) and not df_cache.empty:
            print("🔄 Usando cache existente debido a error de descarga")
            df_cache = pd.read_parquet(cache_path)
            # Asegurar timezone en cache
            if df_cache["fecha"].dt.tz is None:
                df_cache["fecha"] = df_cache["fecha"].dt.tz_localize('UTC')
        else:
            raise e

    if integrity_check and not df_cache.empty:
        print("Verificando integridad del dataset...")
        before = len(df_cache)
        df_cache = df_cache.drop_duplicates(subset="fecha", keep="last")
        df_cache = df_cache.sort_values("fecha").reset_index(drop=True)
        df_cache = df_cache.dropna(subset=["fecha", "close"])
        df_cache["close"] = pd.to_numeric(df_cache["close"], errors="coerce")
        df_cache = df_cache.dropna(subset=["close"])
        df_cache = df_cache[df_cache["close"] > 0]
        after = len(df_cache)
        if after < before:
            print(f"Limpieza completada. {before - after} registros corruptos/duplicados eliminados.")
            log_gridbot_event(f"Limpieza: {before - after} registros corruptos o duplicados eliminados.", log_dir, log_filename)

    # Guardar cache
    df_cache.to_parquet(cache_path, index=False, compression="snappy")
    print(f"Cache actualizado → {cache_path} ({len(df_cache)} registros totales).")
    log_gridbot_event(f"Cache actualizado con {len(df_cache)} registros.", log_dir, log_filename)

    # Filtrar por rango solicitado  
    df_final = df_cache[(df_cache["fecha"] >= fecha_inicio) & (df_cache["fecha"] <= fecha_fin)]
    
    # Verificar integridad de fechas
    df_final = verificar_integridad_fechas(df_final)
    
    print(f"Datos finales cargados: {len(df_final)} registros en rango solicitado.\n")
    log_gridbot_event(f"Finalizada carga incremental. Datos cargados: {len(df_final)} registros.\n", log_dir, log_filename)
    
    return df_final