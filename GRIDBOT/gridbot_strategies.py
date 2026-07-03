# src/gridbot_strategies.py
import pandas as pd
import numpy as np
from gridbot_core import inicializar_gridbot
from gridbot_core import simular_gridbot
from gridbot_trades import generar_detalle_trades_nivel, calcular_evolucion_balance

def personalizacion_gridbot_inyeccion(
    df,
    estado_inicial,
    monto_inyeccion=100.0,
    frecuencia="mensual",
    fee_pct=0.05,
    n_down=3,
    n_up=3,
    tolerancia_pct=0.01,
    mostrar_barra=True
):
    global precio_corte
    if "fecha" not in df.columns or "close" not in df.columns:
        raise ValueError("El DataFrame debe contener columnas 'fecha' y 'close'.")
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha", "close"]).sort_values("fecha").reset_index(drop=True)
    if frecuencia == "mensual":
        freq = "M"
    elif frecuencia == "semanal":
        freq = "W"
    elif frecuencia == "trimestral":
        freq = "Q"
    else:
        raise ValueError("Frecuencia inválida. Usa: 'mensual', 'semanal' o 'trimestral'.")
    df["periodo"] = df["fecha"].dt.to_period(freq)
    bloques = [grupo for _, grupo in df.groupby("periodo")]
    ordenes_global, trades_global, balance_global = [], [], pd.DataFrame()
    print(f"\n=== 🚀 Iniciando simulación con inyección cada {frecuencia} ===")
    ultimo_balance = {
        "balance_ETH": estado_inicial["distribucion_capital"]["balance_ETH"],
        "balance_USDT": estado_inicial["distribucion_capital"]["balance_USDT"],
        "valor_total_USDT": (
            estado_inicial["distribucion_capital"]["balance_USDT"]
            + estado_inicial["distribucion_capital"]["balance_ETH"]
            * estado_inicial.get("precio_inicial", df["close"].iloc[0])
        )
    }
    rango_base_inferior = estado_inicial["niveles"]["niveles_totales"][0]
    rango_base_superior = estado_inicial["niveles"]["niveles_totales"][-1]
    for idx, df_bloque in enumerate(bloques):
        fecha_inicio_bloque = df_bloque["fecha"].min().normalize()
        print(f"\n--- 🧩 Bloque {idx+1}/{len(bloques)}: {fecha_inicio_bloque.date()} ---")
        if idx > 0:
            if fecha_inicio_bloque in df_bloque["fecha"].values:
                precio_corte = df_bloque.loc[df_bloque["fecha"] == fecha_inicio_bloque, "close"].values[0]
            else:
                df_aux = df_bloque.sort_values("fecha")
                precio_corte = np.interp(
                    fecha_inicio_bloque.timestamp(),
                    df_aux["fecha"].astype(np.int64) / 1e9,
                    df_aux["close"]
                )
            if not np.isfinite(precio_corte):
                raise ValueError(f"No se pudo determinar el precio en {fecha_inicio_bloque.date()}")
            balance_eth = ultimo_balance["balance_ETH"]
            balance_usdt = ultimo_balance["balance_USDT"]
            pct_eth = estado_inicial["distribucion_capital"]["porcent_ETH"] / 100
            usd_para_eth = monto_inyeccion * pct_eth
            eth_comprado = usd_para_eth / precio_corte
            fee_eth = eth_comprado * (fee_pct / 100)
            eth_neto = eth_comprado - fee_eth
            fee_usdt_equiv = fee_eth * precio_corte
            balance_eth += eth_neto
            balance_usdt += monto_inyeccion * (1 - pct_eth)
            valor_total = balance_usdt + (balance_eth * precio_corte)
            balance_usdt = max(balance_usdt, 0.0)
            precio_superior = rango_base_superior
            precio_inferior = rango_base_inferior
            if not (precio_inferior <= precio_corte <= precio_superior):
                raise ValueError(f"El precio de corte {precio_corte} está fuera del rango [{precio_inferior}, {precio_superior}].")
            nuevo_estado = inicializar_gridbot(
                df=pd.DataFrame([{"close": precio_corte}]),
                precio_superior=precio_superior,
                precio_inferior=precio_inferior,
                gap_pct=estado_inicial["niveles"]["gap_pct"],
                balance_inicial=valor_total,
                modo=estado_inicial["montos_por_grid"]["modo"]
            )
            nuevo_estado["distribucion_capital"]["balance_ETH"] = balance_eth
            nuevo_estado["distribucion_capital"]["balance_USDT"] = balance_usdt
            nuevo_estado["distribucion_capital"]["fee_inyeccion_USDT"] = fee_usdt_equiv
            estado_inicial = nuevo_estado
        ordenes = simular_gridbot(df_bloque, estado_inicial, n_down, n_up, tolerancia_pct, mostrar_barra)
        detalle_trades = generar_detalle_trades_nivel(ordenes, estado_inicial, fee_pct)
        evolucion_balance = calcular_evolucion_balance(ordenes, estado_inicial, df_bloque, fee_pct)
        evolucion_balance["valor_total_USDT"] = (
            evolucion_balance["balance_USDT"]
            + evolucion_balance["balance_ETH"] * evolucion_balance["price"]
        )
        if idx > 0:
            mask_start = (evolucion_balance["tick"] == 0) & (evolucion_balance["side"] == "start")
            evolucion_balance.loc[mask_start, "comision_USDT"] = estado_inicial["distribucion_capital"].get(
                "fee_inyeccion_USDT", 0
            )
        ordenes_global.extend(ordenes)
        trades_global.append(detalle_trades)
        balance_global = pd.concat([balance_global, evolucion_balance], ignore_index=True)
        ultimo_balance = evolucion_balance.iloc[-1].to_dict()
    df_trades_final = pd.concat(trades_global, ignore_index=True)
    capital_final = ultimo_balance["valor_total_USDT"]
    print("\n=== 🏁 Simulación con inyecciones completada exitosamente ===")
    print(f"Total de bloques simulados: {len(bloques)}")
    print(f"Capital final total: {capital_final:.2f} USDT")
    return {
        "ordenes": ordenes_global,
        "detalle_trades": df_trades_final,
        "evolucion_balance": balance_global,
        "estado_final": estado_inicial
    }

def generar_rango_dinamico(df_bloque, estado_inicial, idx, precio_actual, precio_predicho):
    """
    Genera un rango dinámico (inferior y superior) según la tendencia estimada por IA.
    Si la predicción indica alza → rango amplio; si indica baja → rango conservador.
    
    CORRECCIÓN: Maneja tanto diccionarios como valores numéricos para precio_predicho
    """
    # EXTRAER el valor numérico del precio predicho
    if isinstance(precio_predicho, dict):
        # Si es diccionario, usar pred_close
        precio_predicho_valor = precio_predicho.get("pred_close", precio_actual)
        tendencia = precio_predicho.get("tendencia", "neutral")
    else:
        # Si ya es numérico, usarlo directamente
        precio_predicho_valor = precio_predicho
        tendencia = "alcista" if precio_predicho_valor > precio_actual else "bajista"
    
    # Lógica de rangos basada en la tendencia REAL del diccionario
    if tendencia == "alcista":
        precio_inferior = precio_actual * (1 - 0.60)   # -60%
        precio_superior = precio_actual * (1 + 4.00)   # +400%
    elif tendencia == "bajista":
        precio_inferior = precio_actual * (1 - 0.90)   # -90%
        precio_superior = precio_actual * (1 + 1.20)   # +120%
    else:
        # Tendencia neutral o desconocida
        precio_inferior = precio_actual * (1 - 0.75)   # -75%
        precio_superior = precio_actual * (1 + 2.50)   # +250%

    print(f"📊 Tendencia detectada: {tendencia.upper()} | Precio actual: {precio_actual:.2f} | Predicho: {precio_predicho_valor:.2f}")
    print(f"→ Nuevo rango dinámico aplicado: Inferior={precio_inferior:.2f}, Superior={precio_superior:.2f}")

    return precio_inferior, precio_superior


def personalizacion_gridbot_inyeccion2(
    df,
    estado_inicial,
    monto_inyeccion=100.0,
    frecuencia="mensual",
    fee_pct=0.05,
    n_down=3,
    n_up=3,
    tolerancia_pct=0.01,
    mostrar_barra=True,
    funcion_rango=None,
    precios_predichos=None
):
    global precio_corte
    if "fecha" not in df.columns or "close" not in df.columns:
        raise ValueError("El DataFrame debe contener columnas 'fecha' y 'close'.")
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha", "close"]).sort_values("fecha").reset_index(drop=True)
    if frecuencia == "mensual":
        freq = "M"
    elif frecuencia == "semanal":
        freq = "W"
    elif frecuencia == "trimestral":
        freq = "Q"
    else:
        raise ValueError("Frecuencia inválida. Usa: 'mensual', 'semanal' o 'trimestral'.")
    df["periodo"] = df["fecha"].dt.to_period(freq)
    bloques = [grupo for _, grupo in df.groupby("periodo")]
    print(f"\n=== 🚀 Iniciando simulación con inyección {frecuencia} y rango IA trimestral ===")
    ultimo_balance = {
        "balance_ETH": estado_inicial["distribucion_capital"]["balance_ETH"],
        "balance_USDT": estado_inicial["distribucion_capital"]["balance_USDT"],
        "valor_total_USDT": (
            estado_inicial["distribucion_capital"]["balance_USDT"]
            + estado_inicial["distribucion_capital"]["balance_ETH"]
            * estado_inicial.get("precio_inicial", df["close"].iloc[0])
        )
    }
    precio_inicial = df["close"].iloc[0]
    if precios_predichos and 0 in precios_predichos and funcion_rango is not None:
        precio_predicho_0 = precios_predichos[0]
        rango_actual_inferior, rango_actual_superior = funcion_rango(df, estado_inicial, 0, precio_inicial, precio_predicho_0)
    else:
        rango_actual_inferior = estado_inicial["niveles"]["niveles_totales"][0]
        rango_actual_superior = estado_inicial["niveles"]["niveles_totales"][-1]
    ordenes_global, trades_global, balance_global = [], [], pd.DataFrame()
    for idx, df_bloque in enumerate(bloques):
        fecha_inicio_bloque = df_bloque["fecha"].min().normalize()
        print(f"\n--- 🧩 Bloque {idx+1}/{len(bloques)}: {fecha_inicio_bloque.date()} ---")
        if fecha_inicio_bloque in df_bloque["fecha"].values:
            precio_corte = df_bloque.loc[df_bloque["fecha"] == fecha_inicio_bloque, "close"].values[0]
        else:
            df_aux = df_bloque.sort_values("fecha")
            precio_corte = np.interp(
                fecha_inicio_bloque.timestamp(),
                df_aux["fecha"].astype(np.int64) / 1e9,
                df_aux["close"]
            )
        if not np.isfinite(precio_corte):
            raise ValueError(f"No se pudo determinar el precio en {fecha_inicio_bloque.date()}")
        if (idx % 3 == 0) and (precios_predichos is not None) and (idx in precios_predichos):
            precio_predicho = precios_predichos[idx]
            rango_actual_inferior, rango_actual_superior = funcion_rango(
                df_bloque, estado_inicial, idx, precio_corte, precio_predicho
            )
        if not (rango_actual_inferior <= precio_corte <= rango_actual_superior):
            raise ValueError(
                f"❌ El precio de corte {precio_corte:.2f} está fuera del rango actual "
                f"[{rango_actual_inferior:.2f}, {rango_actual_superior:.2f}]. "
                f"Ajusta los niveles o revisa la predicción IA."
            )
        if idx > 0:
            balance_eth = ultimo_balance["balance_ETH"]
            balance_usdt = ultimo_balance["balance_USDT"]
            pct_eth = estado_inicial["distribucion_capital"]["porcent_ETH"] / 100
            usd_para_eth = monto_inyeccion * pct_eth
            eth_comprado = usd_para_eth / precio_corte
            fee_eth = eth_comprado * (fee_pct / 100)
            eth_neto = eth_comprado - fee_eth
            fee_usdt_equiv = fee_eth * precio_corte
            balance_eth += eth_neto
            balance_usdt += monto_inyeccion * (1 - pct_eth)
            valor_total = balance_usdt + (balance_eth * precio_corte)
            nuevo_estado = inicializar_gridbot(
                df=pd.DataFrame([{"close": precio_corte}]),
                precio_superior=rango_actual_superior,
                precio_inferior=rango_actual_inferior,
                gap_pct=estado_inicial["niveles"]["gap_pct"],
                balance_inicial=valor_total,
                modo=estado_inicial["montos_por_grid"]["modo"]
            )
            nuevo_estado["distribucion_capital"]["balance_ETH"] = balance_eth
            nuevo_estado["distribucion_capital"]["balance_USDT"] = balance_usdt
            nuevo_estado["distribucion_capital"]["fee_inyeccion_USDT"] = fee_usdt_equiv
            estado_inicial = nuevo_estado
        ordenes = simular_gridbot(df_bloque, estado_inicial, n_down, n_up, tolerancia_pct, mostrar_barra)
        detalle_trades = generar_detalle_trades_nivel(ordenes, estado_inicial, fee_pct)
        evolucion_balance = calcular_evolucion_balance(ordenes, estado_inicial, df_bloque, fee_pct)
        evolucion_balance["valor_total_USDT"] = (
            evolucion_balance["balance_USDT"] + evolucion_balance["balance_ETH"] * evolucion_balance["price"]
        )
        if idx > 0:
            mask_start = (evolucion_balance["tick"] == 0) & (evolucion_balance["side"] == "start")
            evolucion_balance.loc[mask_start, "comision_USDT"] = estado_inicial["distribucion_capital"].get("fee_inyeccion_USDT", 0)
        ordenes_global.extend(ordenes)
        trades_global.append(detalle_trades)
        balance_global = pd.concat([balance_global, evolucion_balance], ignore_index=True)
        ultimo_balance = evolucion_balance.iloc[-1].to_dict()
    df_trades_final = pd.concat(trades_global, ignore_index=True)
    capital_final = ultimo_balance["valor_total_USDT"]
    print("\n=== 🏁 Simulación completada ===")
    print(f"Frecuencia: {frecuencia} | Total de bloques: {len(bloques)} | Capital final: {capital_final:.2f} USDT")
    return {
        "ordenes": ordenes_global,
        "detalle_trades": df_trades_final,
        "evolucion_balance": balance_global,
        "estado_final": estado_inicial
    }

def personalizacion_gridbot_inyeccion_con_ia(
    df,
    estado_inicial,
    monto_inyeccion=100.0,
    frecuencia="mensual",
    fee_pct=0.05,
    n_down=3,
    n_up=3,
    tolerancia_pct=0.01,
    mostrar_barra=True,
    funcion_rango=None,
    precios_predichos=None
):
    """
    Versión de Estrategia 2 (Inyecciones Periódicas) mejorada con IA
    """
    # Usar la función original pero con capacidad de IA
    resultado = personalizacion_gridbot_inyeccion(
        df=df,
        estado_inicial=estado_inicial,
        monto_inyeccion=monto_inyeccion,
        frecuencia=frecuencia,
        fee_pct=fee_pct,
        n_down=n_down,
        n_up=n_up,
        tolerancia_pct=tolerancia_pct,
        mostrar_barra=mostrar_barra
    )
    
    return resultado