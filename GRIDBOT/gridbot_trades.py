# src/gridbot_trades.py
import pandas as pd

def generar_detalle_trades_nivel_ETH(ordenes, fee_pct=0.1):
    """Genera detalle de trades para modo ETH"""
    df = pd.DataFrame(ordenes)
    if df.empty:
        return pd.DataFrame(columns=[
            '#', 'Estado', 'Fecha Compra', 'Precio Compra', 'Volumen Compra (USDT)',
            'Comisión Compra (USDT)', 'Fecha Venta', 'Precio Venta',
            'Volumen Venta (USDT)', 'Comisión Venta (USDT)', 'Profit Neto (USDT)', 'Nota'
        ])
    df = df[df["status"] == "executed"].copy()
    df = df[df["side"].isin(["start", "buy", "sell"])].copy()
    trades = []
    trade_id = 1
    compra_inicial = None
    posiciones_abiertas = []
    ordenes_dict = df.to_dict(orient="records")
    for o in ordenes_dict:
        o["fecha"] = f"Tick {o['tick']}"
        o["comision"] = (o["price"] * o["amount"]) * (fee_pct / 100.0)
        if o["side"] == "start":
            compra_inicial = o
            trades.append({
                "#": trade_id,
                "Estado": "Inicial (setup)",
                "Fecha Compra": o["fecha"],
                "Precio Compra": round(o["price"], 6),
                "Volumen Compra (USDT)": "--",
                "Comisión Compra (USDT)": "--",
                "Fecha Venta": "--",
                "Precio Venta": "--",
                "Volumen Venta (USDT)": "--",
                "Comisión Venta (USDT)": "--",
                "Profit Neto (USDT)": "--",
                "Nota": "Compra inicial al arrancar la inversión. (Modo ETH)"
            })
            trade_id += 1
            continue
        if o["side"] == "buy":
            posiciones_abiertas.append(o)
        elif o["side"] == "sell":
            buy_match = None
            for b in reversed(posiciones_abiertas):
                if b["price"] < o["price"]:
                    buy_match = b
                    break
            if buy_match is not None:
                posiciones_abiertas = [x for x in posiciones_abiertas if x is not buy_match]
                volumen_buy = buy_match["price"] * buy_match["amount"]
                volumen_sell = o["price"] * o["amount"]
                comision_buy = buy_match["comision"]
                comision_sell = o["comision"]
                profit_neto = (volumen_sell - volumen_buy) - (comision_buy + comision_sell)
                trades.append({
                    "#": trade_id,
                    "Estado": "Completado",
                    "Fecha Compra": buy_match["fecha"],
                    "Precio Compra": round(buy_match["price"], 6),
                    "Volumen Compra (USDT)": round(volumen_buy, 6),
                    "Comisión Compra (USDT)": round(comision_buy, 6),
                    "Fecha Venta": o["fecha"],
                    "Precio Venta": round(o["price"], 6),
                    "Volumen Venta (USDT)": round(volumen_sell, 6),
                    "Comisión Venta (USDT)": round(comision_sell, 6),
                    "Profit Neto (USDT)": round(profit_neto, 6),
                    "Nota": f"Venta {o['price']} ↔ compra {buy_match['price']} (ciclo normal)."
                })
                trade_id += 1
            elif compra_inicial is not None and compra_inicial["price"] < o["price"]:
                volumen_buy = compra_inicial["price"] * compra_inicial["amount"]
                volumen_sell = o["price"] * o["amount"]
                comision_buy = compra_inicial["comision"]
                comision_sell = o["comision"]
                profit_neto = (volumen_sell - volumen_buy) - (comision_buy + comision_sell)
                trades.append({
                    "#": trade_id,
                    "Estado": "Completado",
                    "Fecha Compra": compra_inicial["fecha"],
                    "Precio Compra": round(compra_inicial["price"], 6),
                    "Volumen Compra (USDT)": round(volumen_buy, 6),
                    "Comisión Compra (USDT)": round(comision_buy, 6),
                    "Fecha Venta": o["fecha"],
                    "Precio Venta": round(o["price"], 6),
                    "Volumen Venta (USDT)": round(volumen_sell, 6),
                    "Comisión Venta (USDT)": round(comision_sell, 6),
                    "Profit Neto (USDT)": round(profit_neto, 6),
                    "Nota": f"Venta {o['price']} ↔ compra inicial {compra_inicial['price']}."
                })
                trade_id += 1
    for b in posiciones_abiertas:
        trades.append({
            "#": trade_id,
            "Estado": "Pendiente para vender",
            "Fecha Compra": b["fecha"],
            "Precio Compra": round(b["price"], 6),
            "Volumen Compra (USDT)": round(b["price"] * b["amount"], 6),
            "Comisión Compra (USDT)": round(b["price"] * b["amount"] * (fee_pct / 100.0), 6),
            "Fecha Venta": "--",
            "Precio Venta": "--",
            "Volumen Venta (USDT)": "--",
            "Comisión Venta (USDT)": "--",
            "Profit Neto (USDT)": "--",
            "Nota": "Compra sin venta ejecutada aún."
        })
        trade_id += 1
    return pd.DataFrame(trades).sort_values(by="#").reset_index(drop=True)


def generar_detalle_trades_USDT(ordenes, fee_pct=0.1):
    """Genera detalle de trades para modo USDT"""
    df = pd.DataFrame(ordenes)
    if df.empty:
        return pd.DataFrame(columns=[
            '#', 'Estado', 'Fecha Compra', 'Precio Compra', 'Volumen Compra (USDT)',
            'Comisión Compra (USDT)', 'Fecha Venta', 'Precio Venta',
            'Volumen Venta (USDT)', 'Comisión Venta (USDT)', 'Profit Neto (USDT)', 'Nota'
        ])
    df = df[df["status"] == "executed"].copy()
    df = df[df["side"].isin(["start", "buy", "sell"])].copy()
    trades = []
    trade_id = 1
    compra_inicial = None
    posiciones_abiertas = []
    ordenes_dict = df.to_dict(orient="records")
    for o in ordenes_dict:
        o["fecha"] = f"Tick {o['tick']}"
        o["comision"] = (o["price"] * o["amount"]) * (fee_pct / 100.0)
        if o["side"] == "start":
            compra_inicial = o
            trades.append({
                "#": trade_id,
                "Estado": "Inicial (setup)",
                "Fecha Compra": o["fecha"],
                "Precio Compra": round(o["price"], 6),
                "Volumen Compra (USDT)": "--",
                "Comisión Compra (USDT)": "--",
                "Fecha Venta": "--",
                "Precio Venta": "--",
                "Volumen Venta (USDT)": "--",
                "Comisión Venta (USDT)": "--",
                "Profit Neto (USDT)": "--",
                "Nota": "Compra inicial al arrancar la inversión. (Modo USDT)"
            })
            trade_id += 1
            continue
        if o["side"] == "buy":
            posiciones_abiertas.append(o)
        elif o["side"] == "sell":
            buy_match = None
            for b in reversed(posiciones_abiertas):
                if b["price"] < o["price"]:
                    buy_match = b
                    break
            if buy_match is not None:
                posiciones_abiertas = [x for x in posiciones_abiertas if x is not buy_match]
                eth_cantidad = buy_match["amount"]
                volumen_buy = buy_match["price"] * eth_cantidad
                volumen_sell = o["price"] * eth_cantidad
                comision_buy = buy_match["comision"]
                comision_sell = o["comision"]
                profit_neto = (volumen_sell - volumen_buy) - (comision_buy + comision_sell)
                trades.append({
                    "#": trade_id,
                    "Estado": "Completado",
                    "Fecha Compra": buy_match["fecha"],
                    "Precio Compra": round(buy_match["price"], 6),
                    "Volumen Compra (USDT)": round(volumen_buy, 6),
                    "Comisión Compra (USDT)": round(comision_buy, 6),
                    "Fecha Venta": o["fecha"],
                    "Precio Venta": round(o["price"], 6),
                    "Volumen Venta (USDT)": round(volumen_sell, 6),
                    "Comisión Venta (USDT)": round(comision_sell, 6),
                    "Profit Neto (USDT)": round(profit_neto, 6),
                    "Nota": f"Venta {o['price']} ↔ compra {buy_match['price']} (ciclo normal)."
                })
                trade_id += 1
            elif compra_inicial is not None and compra_inicial["price"] < o["price"]:
                eth_cantidad = compra_inicial["amount"]
                volumen_buy = compra_inicial["price"] * eth_cantidad
                volumen_sell = o["price"] * eth_cantidad
                comision_buy = compra_inicial["comision"]
                comision_sell = o["comision"]
                profit_neto = (volumen_sell - volumen_buy) - (comision_buy + comision_sell)
                trades.append({
                    "#": trade_id,
                    "Estado": "Completado",
                    "Fecha Compra": compra_inicial["fecha"],
                    "Precio Compra": round(compra_inicial["price"], 6),
                    "Volumen Compra (USDT)": round(volumen_buy, 6),
                    "Comisión Compra (USDT)": round(comision_buy, 6),
                    "Fecha Venta": o["fecha"],
                    "Precio Venta": round(o["price"], 6),
                    "Volumen Venta (USDT)": round(volumen_sell, 6),
                    "Comisión Venta (USDT)": round(comision_sell, 6),
                    "Profit Neto (USDT)": round(profit_neto, 6),
                    "Nota": f"Venta {o['price']} ↔ compra inicial {compra_inicial['price']}."
                })
                trade_id += 1
    for b in posiciones_abiertas:
        trades.append({
            "#": trade_id,
            "Estado": "Pendiente para vender",
            "Fecha Compra": b["fecha"],
            "Precio Compra": round(b["price"], 6),
            "Volumen Compra (USDT)": round(b["price"] * b["amount"], 6),
            "Comisión Compra (USDT)": round(b["price"] * b["amount"] * (fee_pct / 100.0), 6),
            "Fecha Venta": "--",
            "Precio Venta": "--",
            "Volumen Venta (USDT)": "--",
            "Comisión Venta (USDT)": "--",
            "Profit Neto (USDT)": "--",
            "Nota": "Compra sin venta ejecutada aún."
        })
        trade_id += 1
    return pd.DataFrame(trades).sort_values(by="#").reset_index(drop=True)


def generar_detalle_trades_nivel(ordenes, estado_inicial, fee_pct=0.1):
    """Router automático según modo ETH/USDT"""
    modo = estado_inicial["montos_por_grid"]["modo"].upper()
    if modo == "ETH":
        return generar_detalle_trades_nivel_ETH(ordenes, fee_pct)
    elif modo == "USDT":
        return generar_detalle_trades_USDT(ordenes, fee_pct)
    else:
        raise ValueError(f"Modo desconocido: {modo}. Debe ser 'ETH' o 'USDT'.")


def calcular_evolucion_balance_ETH(ordenes, estado_inicial, df_precios=None, fee_pct=0.0):
    """Evolución de balance en modo ETH"""
    balance_eth = estado_inicial["distribucion_capital"]["balance_ETH"]
    balance_usdt = estado_inicial["distribucion_capital"]["balance_USDT"]
    df_ordenes = pd.DataFrame(ordenes)
    df_ordenes = df_ordenes[df_ordenes["status"] == "executed"].copy()
    df_ordenes = df_ordenes.sort_values(by=["tick", "side"]).reset_index(drop=True)
    historial = []
    for _, o in df_ordenes.iterrows():
        side = o["side"].lower()
        price = float(o["price"])
        amount = float(o["amount"])
        tick = int(o["tick"])
        if side == "start":
            comision = 0.0
        else:
            comision = (price * amount) * (fee_pct / 100.0)
        if side == "start":
            if fee_pct > 0:
                balance_eth = balance_eth - balance_eth * (fee_pct / 100.0)
                comision = (balance_eth * (fee_pct / 100.0)) * price
        elif side == "buy":
            costo_usdt = price * amount
            balance_usdt -= costo_usdt
            balance_eth += amount - (comision / price)
        elif side == "sell":
            ingreso_usdt = (price * amount) - comision
            balance_usdt += ingreso_usdt
            balance_eth -= amount
        precio_ref = (
            df_precios.iloc[tick]["close"]
            if df_precios is not None and tick < len(df_precios)
            else price
        )
        valor_total_usdt = balance_usdt + balance_eth * precio_ref
        historial.append({
            "fecha": o["fecha"] if "fecha" in o else f"Tick {tick}",
            "tick": tick,
            "side": side,
            "price": round(price, 6),
            "amount_ETH": round(amount, 6),
            "comision_USDT": round(comision, 6),
            "balance_ETH": round(balance_eth, 6),
            "balance_USDT": round(balance_usdt, 6),
            "valor_total_USDT": round(valor_total_usdt, 6)
        })
    return pd.DataFrame(historial)


def calcular_evolucion_balance_USDT(ordenes, estado_inicial, df_precios=None, fee_pct=0.0):
    """Evolución de balance en modo USDT"""
    balance_eth = estado_inicial["distribucion_capital"]["balance_ETH"]
    balance_usdt = estado_inicial["distribucion_capital"]["balance_USDT"]
    df_ordenes = pd.DataFrame(ordenes)
    df_ordenes = df_ordenes[df_ordenes["status"] == "executed"].copy()
    df_ordenes = df_ordenes.sort_values(by=["tick", "side"]).reset_index(drop=True)
    historial = []
    for _, o in df_ordenes.iterrows():
        side = o["side"].lower()
        price = float(o["price"])
        amount_usdt = float(o["amount"] * price)  # amount viene en USDT
        tick = int(o["tick"])
        if side == "start":
            comision = 0.0
        else:
            comision = amount_usdt * (fee_pct / 100.0)
        if side == "start":
            if fee_pct > 0:
                balance_eth = balance_eth - balance_eth * (fee_pct / 100.0)
                comision = (balance_eth * (fee_pct / 100.0)) * price
        elif side == "buy":
            costo_usdt = amount_usdt - comision
            balance_usdt -= costo_usdt
            balance_eth += costo_usdt / price
        elif side == "sell":
            ingreso_usdt = amount_usdt - comision
            balance_usdt += ingreso_usdt
            balance_eth -= ingreso_usdt / price
        precio_ref = (
            df_precios.iloc[tick]["close"]
            if df_precios is not None and tick < len(df_precios)
            else price
        )
        valor_total_usdt = balance_usdt + balance_eth * precio_ref
        historial.append({
            "fecha": o["fecha"] if "fecha" in o else f"Tick {tick}",
            "tick": tick,
            "side": side,
            "price": round(price, 6),
            "amount_usdt": round(amount_usdt, 6),
            "comision_USDT": round(comision, 6),
            "balance_ETH": round(balance_eth, 6),
            "balance_USDT": round(balance_usdt, 6),
            "valor_total_USDT": round(valor_total_usdt, 6)
        })
    return pd.DataFrame(historial)


def calcular_evolucion_balance(ordenes, estado_inicial, df_precios=None, fee_pct=0.0):
    """Router automático según modo ETH/USDT"""
    modo = estado_inicial["montos_por_grid"]["modo"].upper()
    if modo == "ETH":
        return calcular_evolucion_balance_ETH(ordenes, estado_inicial, df_precios, fee_pct)
    elif modo == "USDT":
        return calcular_evolucion_balance_USDT(ordenes, estado_inicial, df_precios, fee_pct)
    else:
        raise ValueError(f"Modo desconocido: {modo}. Debe ser 'ETH' o 'USDT'.")