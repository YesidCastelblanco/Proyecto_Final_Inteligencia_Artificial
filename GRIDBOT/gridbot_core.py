# src/gridbot_core.py
import numpy as np

def distribuir_capital(precio_inicial, precio_superior, precio_inferior, balance_inicial):
    if precio_superior == precio_inferior:
        raise ValueError("El precio superior e inferior no pueden ser iguales.")
    if precio_inicial < precio_inferior or precio_inicial > precio_superior:
        raise ValueError("El precio inicial debe estar dentro del rango definido.")
    if balance_inicial <= 0:
        raise ValueError("El balance en USDT debe ser mayor que 0.")
    if precio_inferior < 0 or precio_superior < 0:
        raise ValueError("Los precios no pueden ser negativos.")
    if precio_inferior >= precio_superior:
        raise ValueError("El precio inferior debe ser menor que el precio superior.")
    if precio_superior == precio_inicial:
        raise ValueError("El precio superior no puede ser igual al precio inicial.")
    if precio_inferior == precio_inicial:
        raise ValueError("El precio inferior no puede ser igual al precio inicial.")
   
    p_eth = (precio_superior - precio_inicial) / (precio_superior - precio_inferior)
    p_eth = max(0, min(1, p_eth))
   
    balance_eth = (balance_inicial * p_eth) / precio_inicial
    balance_usdt_restante = balance_inicial * (1 - p_eth)
   
    return {
        "precio_inicial": precio_inicial,
        "porcent_ETH": round(p_eth * 100, 2),
        "porcent_USDT": round((1 - p_eth) * 100, 2),
        "balance_ETH": balance_eth,
        "balance_USDT": balance_usdt_restante
    }

def generar_grilla(precio_inicial, precio_superior, precio_inferior, gap_pct):
    if precio_superior == precio_inferior:
        raise ValueError("El precio superior e inferior no pueden ser iguales.")
    if precio_inicial < precio_inferior or precio_inicial > precio_superior:
        raise ValueError("El precio inicial debe estar dentro del rango definido.")
    if precio_inferior < 0 or precio_superior < 0 or precio_inicial < 0:
        raise ValueError("Los precios no pueden ser negativos.")
    if precio_inferior >= precio_superior:
        raise ValueError("El precio inferior debe ser menor que el precio superior.")
    if precio_superior == precio_inicial:
        raise ValueError("El precio superior no puede ser igual al precio inicial.")
    if precio_inferior == precio_inicial:
        raise ValueError("El precio inferior no puede ser igual al precio inicial.")
    if gap_pct <= 0:
        raise ValueError("El gap porcentual debe ser mayor que 0.")
   
    gap = gap_pct / 100.0
    niveles_up, niveles_down = [], []
    k = 1
    while True:
        nivel = precio_inicial * (1 + gap)**k
        if nivel <= precio_superior:
            niveles_up.append(round(nivel, 3))
            k += 1
        else:
            break
    k = 1
    while True:
        nivel = precio_inicial * (1 + gap)**(-k)
        if nivel >= precio_inferior:
            niveles_down.append(round(nivel, 3))
            k += 1
        else:
            break
   
    nivelesT = sorted(niveles_down[::-1] + [precio_inicial] + niveles_up)
    return {
        "niveles_totales": nivelesT,
        "niveles_up": niveles_up,
        "niveles_down": niveles_down,
        "gap_pct": gap_pct
    }

def asignar_montos_por_grid(precio_inicial, niveles_up, niveles_down, balance_inicial, modo="ETH"):
    precio_ref = precio_inicial
    if balance_inicial <= 0:
        raise ValueError("El balance_inicial debe ser > 0.")
    if modo.upper() not in ("ETH", "USDT"):
        raise ValueError('modo debe ser "ETH" o "USDT".')
    total_grids = len(niveles_up) + len(niveles_down)
    if total_grids == 0:
        raise ValueError("No hay niveles de grilla para asignar montos.")
    monto_por_grid_usdt = balance_inicial / total_grids
    if modo.upper() == "ETH":
        if precio_ref is None or precio_ref <= 0:
            raise ValueError("precio_ref (>0) es requerido para modo ETH.")
        monto = float(monto_por_grid_usdt / precio_ref)
    else:
        monto = float(monto_por_grid_usdt)
    return {"modo": modo.upper(), "level_mount_grid": monto}

def inicializar_gridbot(df, precio_superior, precio_inferior, gap_pct, balance_inicial, modo="ETH"):
    precio_inicial = df.iloc[0]['close']
    distribucion = distribuir_capital(precio_inicial, precio_superior, precio_inferior, balance_inicial)
    grilla = generar_grilla(precio_inicial, precio_superior, precio_inferior, gap_pct)
    montos = asignar_montos_por_grid(precio_inicial, grilla["niveles_up"], grilla["niveles_down"], balance_inicial, modo)
    return {
        "precio_inicial": precio_inicial,
        "capital_base": balance_inicial,
        "distribucion_capital": distribucion,
        "niveles": grilla,
        "montos_por_grid": montos
    }

def simular_gridbot(df, estado_inicial, n_down=2, n_up=2, tolerancia_pct=0.1, mostrar_barra=True):
    from tqdm import tqdm
    from bisect import bisect_left, bisect_right
    precios = df["close"].to_numpy(dtype=np.float64)
    n_ticks = len(precios)
    niveles = np.array(estado_inicial["niveles"]["niveles_totales"], dtype=np.float64)
    monto_grid = estado_inicial["montos_por_grid"]["level_mount_grid"]
    modo = estado_inicial["montos_por_grid"]["modo"]
    tol = tolerancia_pct / 100.0
    orders = []
    activos_buy = set()
    activos_sell = set()
    precio_inicial = estado_inicial["precio_inicial"]
    idx_medio = np.abs(niveles - precio_inicial).argmin()
    activos_buy.update(niveles[max(0, idx_medio - n_down):idx_medio])
    activos_sell.update(niveles[idx_medio + 1: idx_medio + 1 + n_up])
    orders.append({
        "fecha": df["fecha"].iloc[0] if "fecha" in df.columns else "Tick 0",
        "tick": 0,
        "side": "start",
        "price": precio_inicial,
        "nivel": precio_inicial,
        "amount": monto_grid / precio_inicial if modo.upper() == "USDT" else monto_grid,
        "status": "executed"
    })
    iterador = tqdm(range(1, n_ticks), total=n_ticks, desc="Simulando GridBot...", ncols=90, disable=not mostrar_barra)
    for i in iterador:
        precio_prev = precios[i - 1]
        precio = precios[i]
        if precio == precio_prev:
            continue
        if precio_prev > precio:
            idx_lo = max(0, bisect_left(niveles, precio * (1 - tol)) - 1)
            idx_hi = min(len(niveles), bisect_right(niveles, precio_prev * (1 + tol)) + 1)
            posibles = niveles[idx_lo:idx_hi]
            for nivel in reversed(posibles):
                if (precio <= nivel <= precio_prev) or abs(nivel - precio) <= nivel * tol or abs(nivel - precio_prev) <= nivel * tol:
                    if nivel in activos_buy:
                        amount = monto_grid / nivel if modo.upper() == "USDT" else monto_grid
                        orders.append({
                            "fecha": df["fecha"].iloc[i] if "fecha" in df.columns else f"Tick {i}",
                            "tick": i,
                            "side": "buy",
                            "price": nivel,
                            "nivel": nivel,
                            "amount": amount,
                            "status": "executed"
                        })
                        activos_buy.remove(nivel)
                        idx = np.where(niveles == nivel)[0][0]
                        if idx + 1 < len(niveles):
                            activos_sell.add(niveles[idx + 1])
        elif precio_prev < precio:
            idx_lo = max(0, bisect_left(niveles, precio_prev * (1 - tol)) - 1)
            idx_hi = min(len(niveles), bisect_right(niveles, precio * (1 + tol)) + 1)
            posibles = niveles[idx_lo:idx_hi]
            for nivel in posibles:
                if (precio_prev <= nivel <= precio) or abs(nivel - precio) <= nivel * tol or abs(nivel - precio_prev) <= nivel * tol:
                    if nivel in activos_sell:
                        amount = monto_grid / nivel if modo.upper() == "USDT" else monto_grid
                        orders.append({
                            "fecha": df["fecha"].iloc[i] if "fecha" in df.columns else f"Tick {i}",
                            "tick": i,
                            "side": "sell",
                            "price": nivel,
                            "nivel": nivel,
                            "amount": amount,
                            "status": "executed"
                        })
                        activos_sell.remove(nivel)
                        idx = np.where(niveles == nivel)[0][0]
                        if idx - 1 >= 0:
                            activos_buy.add(niveles[idx - 1])
        if len(activos_sell) < n_up:
            max_sell = max(activos_sell) if activos_sell else precio
            idx_sell = np.abs(niveles - max_sell).argmin()
            for j in range(1, n_up - len(activos_sell) + 1):
                if idx_sell + j < len(niveles):
                    activos_sell.add(niveles[idx_sell + j])
        if len(activos_buy) < n_down:
            min_buy = min(activos_buy) if activos_buy else precio
            idx_buy = np.abs(niveles - min_buy).argmin()
            for j in range(1, n_down - len(activos_buy) + 1):
                if idx_buy - j >= 0:
                    activos_buy.add(niveles[idx_buy - j])
        if i % 1000 == 0:
            iterador.set_postfix({}, refresh=False)
    iterador.close()
    print("Simulación completada.\n")
    return orders