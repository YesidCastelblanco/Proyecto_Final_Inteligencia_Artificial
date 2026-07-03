# src/utils.py
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Colores para el GridBot
TRADING_COLORS = {
    'buy': '#00C853',      # Verde brillante
    'sell': '#FF3D00',     # Rojo intenso
    'grid': '#FF9800',     # Naranja
    'price': '#2196F3',    # Azul
    'profit': '#4CAF50',   # Verde oscuro
    'loss': '#F44336',     # Rojo oscuro
    'background': '#1E1E1E',
    'panel': '#2E2E2E'
}

def log_gridbot_event(message, log_dir="logs", log_filename="logs_gridbot.txt"):
    """Guarda mensajes con timestamp en un log persistente dentro de /logs."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def plot_gridbot(df, estado_inicial, ordenes, show_level_values=True, show_order_labels=True, show_grid=True):
    """Versión simple del gráfico del GridBot"""
    niveles = estado_inicial["niveles"]["niveles_totales"]
    plt.figure(figsize=(20, 6))
    plt.plot(df.index, df["close"], label="Precio", color="gray", linewidth=1)
    
    if show_grid:
        for nivel in niveles:
            plt.axhline(y=nivel, color="#FF7F0E", linestyle="--", linewidth=0.8,
                        label="Niveles Grid" if nivel == niveles[0] else "")
            if show_level_values:
                plt.text(x=len(df)*1.055, y=nivel, s=f"{nivel:.3f}",
                         color="black", fontsize=7, va='center')
    
    buys = [o for o in ordenes if o["side"] in ("buy", "start")]
    sells = [o for o in ordenes if o["side"] == "sell"]
    offset = (df["close"].max() - df["close"].min()) * 0.02
    
    plt.scatter([o["tick"] for o in buys], [o["nivel"] - offset for o in buys],
                marker="^", color="green", s=70, label="Compras (BUY)", zorder=5)
    plt.scatter([o["tick"] for o in sells], [o["nivel"] + offset for o in sells],
                marker="v", color="red", s=70, label="Ventas (SELL)", zorder=5)
    
    if show_order_labels:
        for o in ordenes:
            if o["side"] in ("buy", "start"):
                plt.annotate(f"{o['nivel']:.2f}", (o["tick"], o["nivel"] - offset),
                             textcoords="offset points", xytext=(0,8), ha='center', fontsize=6, color="blue")
            elif o["side"] == "sell":
                plt.annotate(f"{o['nivel']:.2f}", (o["tick"], o["nivel"] + offset),
                             textcoords="offset points", xytext=(0,-14), ha='center', fontsize=6, color="blue")
    
    plt.title("Simulación GridBot")
    plt.xlabel("Tick")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_gridbot_mejorado(df, estado_inicial, ordenes, evolucion_balance):
    """
    Visualización profesional del GridBot con 4 paneles:
    1. Precios + órdenes + grid
    2. Evolución del valor total
    3. Distribución de balance ETH vs USDT
    4. Drawdown
    """
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor(TRADING_COLORS['background'])
    
    gs = plt.GridSpec(4, 2, figure=fig, height_ratios=[2.5, 1, 1, 1], hspace=0.3)
    
    # === 1. Panel Principal: Precios + Órdenes + Grid ===
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(TRADING_COLORS['panel'])
    
    ax1.plot(df.index, df['close'], label='Precio ETH/USDT', color=TRADING_COLORS['price'], linewidth=1.8, alpha=0.9)
    
    # Niveles del grid
    niveles = estado_inicial["niveles"]["niveles_totales"]
    for i, nivel in enumerate(niveles):
        color = '#FF6B35' if i % 2 == 0 else '#FFA726'
        ax1.axhline(y=nivel, color=color, linestyle='--', alpha=0.7, linewidth=0.9)
    
    # Órdenes
    buys = [o for o in ordenes if o["side"] in ("buy", "start")]
    sells = [o for o in ordenes if o["side"] == "sell"]
    
    ax1.scatter([o["tick"] for o in buys], [o["price"] for o in buys],
                color=TRADING_COLORS['buy'], marker='^', s=100, label='Compras', zorder=5,
                edgecolors='white', linewidth=1)
    ax1.scatter([o["tick"] for o in sells], [o["price"] for o in sells],
                color=TRADING_COLORS['sell'], marker='v', s=100, label='Ventas', zorder=5,
                edgecolors='white', linewidth=1)
    
    ax1.set_title('📈 GridBot IA - Precios y Órdenes de Trading', fontsize=18, fontweight='bold', color='white', pad=30)
    ax1.set_ylabel('Precio (USDT)', fontsize=14, color='white')
    ax1.legend(facecolor=TRADING_COLORS['panel'], edgecolor='white', fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.tick_params(colors='white', labelsize=11)
    
    # === 2. Evolución del Valor Total ===
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(TRADING_COLORS['panel'])
    
    ax2.plot(evolucion_balance.index, evolucion_balance['valor_total_USDT'],
             color=TRADING_COLORS['profit'], linewidth=2.5, label='Valor Total')
    ax2.fill_between(evolucion_balance.index, evolucion_balance['valor_total_USDT'],
                     evolucion_balance['valor_total_USDT'].iloc[0],
                     color=TRADING_COLORS['profit'], alpha=0.2)
    
    ax2.set_title('💰 Evolución del Portafolio', fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel('USDT', color='white')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')
    ax2.legend(facecolor=TRADING_COLORS['panel'], edgecolor='white')
    
    # === 3. Distribución de Balance ===
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(TRADING_COLORS['panel'])
    
    ax3.plot(evolucion_balance.index, evolucion_balance['balance_USDT'],
             label='Balance USDT', color=TRADING_COLORS['price'], linewidth=2)
    ax3.plot(evolucion_balance.index,
             evolucion_balance['balance_ETH'] * evolucion_balance['price'],
             label='Valor ETH (en USDT)', color=TRADING_COLORS['grid'], linewidth=2)
    
    ax3.set_title('🏦 Composición del Balance', fontsize=14, fontweight='bold', color='white')
    ax3.set_ylabel('USDT', color='white')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(colors='white')
    ax3.legend(facecolor=TRADING_COLORS['panel'], edgecolor='white')
    
    # === 4. Drawdown ===
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor(TRADING_COLORS['panel'])
    
    equity = evolucion_balance['valor_total_USDT']
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100
    
    ax4.fill_between(evolucion_balance.index, drawdown, 0,
                     color=TRADING_COLORS['loss'], alpha=0.6)
    ax4.plot(evolucion_balance.index, drawdown, color=TRADING_COLORS['loss'], linewidth=2)
    
    ax4.set_title(f'📉 Drawdown Máximo: {drawdown.min():.2f}%', fontsize=14, fontweight='bold', color='white')
    ax4.set_ylabel('Drawdown (%)', color='white')
    ax4.set_xlabel('Ticks', color='white')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(colors='white')
    
    # === 5. Estadísticas rápidas ===
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    profit_total = equity.iloc[-1] - equity.iloc[0]
    retorno_porcentaje = (profit_total / equity.iloc[0]) * 100
    
    stats_text = f"""
    📊 RESUMEN FINAL DE PERFORMANCE
    ┌────────────────────────────────────────┐
    │ Capital Inicial:     ${equity.iloc[0]:,.2f}          │
    │ Capital Final:       ${equity.iloc[-1]:,.2f}          │
    │ Profit Neto:         ${profit_total:,.2f}          │
    │ Retorno Total:       {retorno_porcentaje:+.2f}%          │
    │ Max Drawdown:        {drawdown.min():.2f}%          │
    │ Trades Totales:      {len(ordenes)}          │
    └────────────────────────────────────────┘
    """
    
    ax5.text(0.5, 0.5, stats_text, transform=ax5.transAxes,
             fontsize=14, fontfamily='monospace', color='white',
             ha='center', va='center', bbox=dict(boxstyle="round,pad=1", facecolor='#2E2E2E', alpha=0.9))
    
    plt.suptitle('🤖 GRIDBOT IA 2025 - Dashboard Profesional', fontsize=20, fontweight='bold', color='white', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()