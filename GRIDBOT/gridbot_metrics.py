# src/gridbot_metrics.py
import pandas as pd
import numpy as np
from tabulate import tabulate

def generar_metricas_gridbot(df_trades, df_balance, risk_free_rate=2.0):
    metricas = {}
    completados = df_trades[df_trades["Estado"] == "Completado"].copy()
    pendientes = df_trades[df_trades["Estado"].str.contains("Pendiente")].copy()
    completados["Profit Neto (USDT)"] = pd.to_numeric(completados["Profit Neto (USDT)"], errors="coerce").fillna(0)
    profit_total = completados["Profit Neto (USDT)"].sum()
    profit_ganadores = completados[completados["Profit Neto (USDT)"] > 0]["Profit Neto (USDT)"].sum()
    profit_perdedores = completados[completados["Profit Neto (USDT)"] < 0]["Profit Neto (USDT)"].sum()
    metricas["Profit Total (USDT)"] = round(profit_total, 4)
    metricas["N° de Trades Completados"] = len(completados)
    metricas["N° de Compras Pendientes"] = len(pendientes)
    metricas["Profit Promedio por Trade (USDT)"] = round(profit_total / max(len(completados), 1), 4)
    n_wins = (completados["Profit Neto (USDT)"] > 0).sum()
    n_losses = (completados["Profit Neto (USDT)"] < 0).sum()
    win_rate = n_wins / max(n_wins + n_losses, 1)
    metricas["Win Rate (%)"] = round(win_rate * 100, 2)
    profit_factor = profit_ganadores / abs(profit_perdedores) if profit_perdedores != 0 else np.inf
    metricas["Profit Factor"] = round(profit_factor, 3)
    if not df_balance.empty:
        equity = df_balance["valor_total_USDT"].astype(float)
        returns = equity.pct_change().fillna(0)
        valor_inicial = equity.iloc[0]
        valor_final = equity.iloc[-1]
        metricas["Valor Inicial (USDT)"] = round(valor_inicial, 4)
        metricas["Valor Final (USDT)"] = round(valor_final, 4)
        metricas["Retorno Total (%)"] = round((valor_final / valor_inicial - 1) * 100, 2)
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        metricas["Max Drawdown (%)"] = round(drawdown.min() * 100, 2)
        metricas["Volatilidad (%)"] = round(returns.std() * 100, 2)
        if "fecha" in df_balance.columns:
            fechas = pd.to_datetime(df_balance["fecha"], errors="coerce").dropna()
            if len(fechas) > 1:
                avg_delta = (fechas.diff().dropna().mean()).total_seconds()
                if avg_delta <= 60:
                    periods_per_year = 525_600
                elif avg_delta <= 3600:
                    periods_per_year = 8_760
                elif avg_delta <= 86400:
                    periods_per_year = 365
                else:
                    periods_per_year = 52
            else:
                periods_per_year = 365
        else:
            periods_per_year = 365
        rf_per_period = (1 + (risk_free_rate / 100)) ** (1 / periods_per_year) - 1
        excess_returns = returns - rf_per_period
        sharpe_ratio = (np.mean(excess_returns) / (np.std(excess_returns) + 1e-9)) * np.sqrt(periods_per_year)
        downside_std = np.std(returns[returns < 0]) + 1e-9
        sortino_ratio = (np.mean(excess_returns) / downside_std) * np.sqrt(periods_per_year)
        metricas["Sharpe Ratio"] = round(sharpe_ratio, 3)
        metricas["Sortino Ratio"] = round(sortino_ratio, 3)
        if len(completados) > 1:
            profits = completados["Profit Neto (USDT)"].values
            diffs = np.sign(profits)
            cambios = np.where(np.diff(diffs) != 0)[0]
            streaks = np.split(diffs, cambios + 1)
            win_streaks = [len(s) for s in streaks if s[0] > 0]
            loss_streaks = [len(s) for s in streaks if s[0] < 0]
            metricas["Máx. Racha Ganadora"] = max(win_streaks) if win_streaks else 0
            metricas["Máx. Racha Perdedoras"] = max(loss_streaks) if loss_streaks else 0
    if "tick" in df_balance.columns:
        ticks_totales = df_balance["tick"].nunique()
        metricas["Ticks Totales"] = ticks_totales
        metricas["Trades por Tick"] = round(len(completados) / max(ticks_totales, 1), 3)
        if "Retorno Total (%)" in metricas:
            metricas["Retorno Medio por Tick (%)"] = round(metricas["Retorno Total (%)"] / max(ticks_totales, 1), 6)
    df_metricas = pd.DataFrame(metricas.items(), columns=["Métrica", "Valor"])
    print("\n=== 📊 MÉTRICAS AVANZADAS GRIDBOT (Sharpe/Sortino dinámicos) ===")
    print(tabulate(df_metricas, headers="keys", tablefmt="github", showindex=False))
    return metricas