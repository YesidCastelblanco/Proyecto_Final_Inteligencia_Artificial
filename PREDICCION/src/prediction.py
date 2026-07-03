def run_full_prediction(frecuencia, horizonte, modelos, fast_mode=True, backtest=False):
    from src.data_loader import load_and_clean_data
    from src.features import safe_features
    from src.models.ensemble import EnsemblePredictor
    from src.volatility.egarch_fhs import simulate_ranges

    # 1. datos
    df_full, df_future = load_and_clean_data()

    # 2. ensemble
    ensemble = EnsemblePredictor(active_models=modelos)
    close_pred = ensemble.predict(
        frecuencia=frecuencia,
        horizonte=horizonte,
        fast_mode=fast_mode,
        run_backtest=backtest
    )

    # 3. rangos EGARCH+FHS
    low_70, high_70, low_90, high_90 = simulate_ranges(close_pred, horizonte)

    # 4. plots (opcional)
    plots = generate_plots(df_full, close_pred, low_70, high_70)

    return {
        "close": int(close_pred),
        "low_70": int(low_70), "high_70": int(high_70),
        "low_90": int(low_90), "high_90": int(high_90),
        "plots": plots
    }