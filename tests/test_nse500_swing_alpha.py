"""
Comprehensive unit tests for NSE500 Swing Alpha Architect.

Tests cover all major modules:
- TamperEvidentLogger
- DataQualityMonitor
- NSE500DataManager
- FeatureEngine (all indicator groups)
- RegimeDetector (HMM)
- DynamicAdaptiveEnsemble
- SignalGenerator
- RiskEngine
- PositionReconciler
- SystemMonitor
- NSE500AlphaArchitect (integration)
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nse500_swing_alpha import (
    KELLY_FRACTION_MAX,
    KELLY_FRACTION_MIN,
    PORTFOLIO_HEAT_LIMIT,
    RISK_PER_TRADE,
    CointegrationAnalyzer,
    ConceptDriftDetector,
    CorporateActionsHandler,
    DataQualityMonitor,
    DrawdownController,
    DynamicAdaptiveEnsemble,
    ExecutionOptimizer,
    FeatureEngine,
    MultiFactorRiskModel,
    NSE500AlphaArchitect,
    NSE500DataManager,
    PerformanceAnalytics,
    PositionReconciler,
    RegimeDetector,
    RegimeState,
    RiskEngine,
    SectorRotationDetector,
    Signal,
    SignalGenerator,
    StateManager,
    StressTestingFramework,
    SystemMonitor,
    TamperEvidentLogger,
    TradeSignal,
    TransactionCost,
    WalkForwardOptimizer,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_ohlcv():
    """Generate a clean OHLCV dataframe for testing."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))

    df = pd.DataFrame(
        {
            "Open": prices * (1 + rng.uniform(-0.005, 0.005, n)),
            "High": prices * (1 + rng.uniform(0.001, 0.02, n)),
            "Low": prices * (1 - rng.uniform(0.001, 0.02, n)),
            "Close": prices,
            "Volume": rng.integers(100000, 5000000, n),
        },
        index=dates,
    )
    return df


@pytest.fixture
def small_ohlcv():
    """Small OHLCV dataset for edge-case testing."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    rng = np.random.default_rng(123)
    prices = np.linspace(100, 120, 30) + rng.normal(0, 1, 30)

    return pd.DataFrame(
        {
            "Open": prices - 0.5,
            "High": prices + 2,
            "Low": prices - 2,
            "Close": prices,
            "Volume": rng.integers(50000, 200000, 30),
        },
        index=dates,
    )


@pytest.fixture
def feature_engine():
    return FeatureEngine()


@pytest.fixture
def risk_engine():
    return RiskEngine(base_capital=300_000)


@pytest.fixture
def data_manager(tmp_path):
    return NSE500DataManager(
        universe_file="NSE500.txt",
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
    )


# ============================================================================
# TAMPER-EVIDENT LOGGER TESTS
# ============================================================================


class TestTamperEvidentLogger:
    def test_log_creates_hash_chain(self):
        logger = TamperEvidentLogger("test_logger")
        h1 = logger.info("First")
        h2 = logger.info("Second")
        assert h1 != h2
        assert len(logger._hash_chain) == 2

    def test_hash_chain_integrity(self):
        logger = TamperEvidentLogger("integrity_test")
        for i in range(10):
            logger.info(f"Entry {i}")
        assert logger.verify_chain()

    def test_unique_hashes(self):
        logger = TamperEvidentLogger("unique_test")
        hashes = [logger.info(f"Message {i}") for i in range(20)]
        assert len(set(hashes)) == 20

    def test_different_levels(self):
        logger = TamperEvidentLogger("level_test")
        h1 = logger.info("info msg")
        h2 = logger.warning("warning msg")
        h3 = logger.error("error msg")
        assert all(h is not None for h in [h1, h2, h3])
        assert len(logger._hash_chain) == 3

    def test_empty_chain_is_valid(self):
        logger = TamperEvidentLogger("empty_test")
        assert logger.verify_chain()

    def test_hash_is_sha256_hex(self):
        logger = TamperEvidentLogger("hash_format")
        h = logger.info("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ============================================================================
# DATA QUALITY MONITOR TESTS
# ============================================================================


class TestDataQualityMonitor:
    def test_ohlcv_valid_data_passes(self):
        """Valid OHLCV data where all constraints hold should pass fully."""
        monitor = DataQualityMonitor()
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "High": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
                "Low": [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                "Close": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "Volume": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            },
            index=dates,
        )
        clean = monitor.validate_ohlcv_consistency(df)
        assert len(clean) == 10

    def test_high_less_than_low_quarantined(self):
        monitor = DataQualityMonitor()
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [90.0, 106.0],  # First row: High < Low
                "Low": [95.0, 96.0],
                "Close": [92.0, 103.0],
                "Volume": [1000, 2000],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="B"),
        )
        clean = monitor.validate_ohlcv_consistency(df)
        assert len(clean) == 1
        assert len(monitor.quarantine) == 1

    def test_negative_volume_quarantined(self):
        monitor = DataQualityMonitor()
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [95.0],
                "Close": [102.0],
                "Volume": [-500],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
        )
        clean = monitor.validate_ohlcv_consistency(df)
        assert len(clean) == 0

    def test_zero_price_quarantined(self):
        monitor = DataQualityMonitor()
        df = pd.DataFrame(
            {
                "Open": [0.0],
                "High": [105.0],
                "Low": [95.0],
                "Close": [102.0],
                "Volume": [1000],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
        )
        clean = monitor.validate_ohlcv_consistency(df)
        assert len(clean) == 0

    def test_close_outside_range_quarantined(self):
        monitor = DataQualityMonitor()
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [95.0],
                "Close": [110.0],  # Above high
                "Volume": [1000],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
        )
        clean = monitor.validate_ohlcv_consistency(df)
        assert len(clean) == 0

    def test_empty_dataframe_passes(self):
        monitor = DataQualityMonitor()
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        clean = monitor.validate_ohlcv_consistency(df)
        assert len(clean) == 0

    def test_time_series_sync_no_gaps(self):
        monitor = DataQualityMonitor()
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"Close": range(10)}, index=dates
        )
        result = monitor.validate_time_series_sync(df)
        assert result["gaps"] == 0
        assert result["coverage_pct"] == 100.0

    def test_time_series_sync_with_gaps(self):
        monitor = DataQualityMonitor()
        # Extend index past the data
        full_dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df_gapped = pd.DataFrame(
            {"Close": [1, 2, 3, 4, 5]},
            index=[full_dates[0], full_dates[1], full_dates[3], full_dates[5], full_dates[9]],
        )
        result = monitor.validate_time_series_sync(df_gapped)
        assert result["gaps"] > 0

    def test_corporate_actions_detection(self):
        monitor = DataQualityMonitor()
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        close = [100, 101, 102, 50, 51, 52, 53, 54, 55, 56]  # 50% drop
        volume = [1000, 1000, 1000, 5000, 1000, 1000, 1000, 1000, 1000, 1000]
        df = pd.DataFrame(
            {
                "Open": close,
                "High": [c + 2 for c in close],
                "Low": [c - 2 for c in close],
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )
        actions = monitor.detect_corporate_actions(df)
        assert len(actions) > 0

    def test_zscore_outlier_detection(self):
        monitor = DataQualityMonitor(z_score_threshold=3.0)
        data = pd.Series([1.0] * 99 + [100.0])  # Single outlier
        outliers = monitor.detect_zscore_outliers(data)
        assert outliers.iloc[-1]  # Last point is outlier

    def test_zscore_no_outliers(self):
        monitor = DataQualityMonitor(z_score_threshold=4.0)
        data = pd.Series(np.random.default_rng(42).normal(0, 1, 100))
        outliers = monitor.detect_zscore_outliers(data)
        # With z_threshold=4 and 100 normal points, expect very few outliers
        assert outliers.sum() <= 2

    def test_full_validation(self, sample_ohlcv):
        monitor = DataQualityMonitor()
        result = monitor.full_validation(sample_ohlcv, "TEST.NS")
        assert result["symbol"] == "TEST.NS"
        assert result["original_rows"] == len(sample_ohlcv)
        assert "time_series" in result
        assert "corporate_actions" in result
        assert "outlier_count" in result

    def test_indicator_stability(self):
        monitor = DataQualityMonitor()
        indicators = pd.DataFrame(
            {
                "stable": np.ones(50),
                "noisy": np.random.default_rng(42).normal(0, 1, 50),
            }
        )
        stability = monitor.validate_indicator_stability(indicators)
        assert "stable" in stability
        assert "noisy" in stability


# ============================================================================
# DATA MANAGER TESTS
# ============================================================================


class TestNSE500DataManager:
    def test_load_universe_mock(self, data_manager):
        symbols = data_manager.load_universe()
        assert len(symbols) == 10
        assert "RELIANCE.NS" in symbols

    def test_load_universe_from_file(self, tmp_path):
        universe_file = tmp_path / "NSE500.txt"
        universe_file.write_text("RELIANCE.NS\nTCS.NS\nINFY.NS\n")
        dm = NSE500DataManager(
            universe_file=str(universe_file),
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
        )
        symbols = dm.load_universe()
        assert symbols == ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

    def test_sha256_computation(self, data_manager):
        data = b"test data for hashing"
        h = data_manager.compute_sha256(data)
        assert len(h) == 64
        # Deterministic
        assert h == data_manager.compute_sha256(data)

    def test_sha256_different_data(self, data_manager):
        h1 = data_manager.compute_sha256(b"data1")
        h2 = data_manager.compute_sha256(b"data2")
        assert h1 != h2

    def test_synthetic_data_generation(self, data_manager):
        df = data_manager._generate_synthetic_data(
            "TEST.NS", "2024-01-01", "2024-06-01"
        )
        assert df is not None
        assert not df.empty
        assert all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"])
        assert (df["High"] >= df["Low"]).all()
        assert (df["Volume"] > 0).all()

    def test_synthetic_data_deterministic(self, data_manager):
        df1 = data_manager._generate_synthetic_data("SAME.NS", "2024-01-01", "2024-03-01")
        df2 = data_manager._generate_synthetic_data("SAME.NS", "2024-01-01", "2024-03-01")
        pd.testing.assert_frame_equal(df1, df2)

    def test_save_and_load_data(self, data_manager):
        df = data_manager._generate_synthetic_data("TEST.NS", "2024-01-01", "2024-03-01")
        data_manager.save_data("TEST.NS", df)
        loaded = data_manager.load_data("TEST.NS")
        assert loaded is not None
        assert len(loaded) == len(df)

    def test_data_integrity_verification(self, data_manager):
        df = data_manager._generate_synthetic_data("INT.NS", "2024-01-01", "2024-03-01")
        data_manager.save_data("INT.NS", df)
        # Should verify okay
        assert data_manager.verify_data_integrity("INT.NS", df)

    def test_data_integrity_violation(self, data_manager):
        df = data_manager._generate_synthetic_data("TAMPER.NS", "2024-01-01", "2024-03-01")
        data_manager.save_data("TAMPER.NS", df)
        # Tamper with data
        tampered = df.copy()
        tampered.iloc[0, 0] = 999999.0
        assert not data_manager.verify_data_integrity("TAMPER.NS", tampered)

    def test_deduplicate(self, data_manager):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"Open": [1, 2, 3, 4, 5], "Close": [1, 2, 3, 4, 5]},
            index=dates,
        )
        # Add duplicate
        dup_df = pd.concat([df, df.iloc[[2]]])
        result = data_manager.deduplicate(dup_df)
        assert len(result) == 5

    def test_get_last_available_date_no_data(self, data_manager):
        result = data_manager.get_last_available_date("NONEXIST.NS")
        assert result is None

    def test_get_last_available_date_with_data(self, data_manager):
        df = data_manager._generate_synthetic_data("DATE.NS", "2024-01-01", "2024-06-01")
        data_manager.save_data("DATE.NS", df)
        result = data_manager.get_last_available_date("DATE.NS")
        assert result is not None


# ============================================================================
# FEATURE ENGINE TESTS
# ============================================================================


class TestFeatureEngineTrend:
    def test_ema_basic(self, feature_engine):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = feature_engine.ema(series, 3)
        assert len(result) == 10
        # EMA should be close to recent values
        assert result.iloc[-1] > result.iloc[0]

    def test_sma_basic(self, feature_engine):
        series = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = feature_engine.sma(series, 3)
        assert result.iloc[2] == 2.0  # (1+2+3)/3
        assert result.iloc[3] == 3.0  # (2+3+4)/3
        assert result.iloc[4] == 4.0  # (3+4+5)/3

    def test_wma_basic(self, feature_engine):
        series = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = feature_engine.wma(series, 3)
        # WMA[2] = (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6
        assert abs(result.iloc[2] - 14 / 6) < 0.001

    def test_dema(self, feature_engine):
        series = pd.Series(np.linspace(1, 50, 50), dtype=float)
        result = feature_engine.dema(series, 10)
        assert len(result) == 50
        # DEMA should track faster than EMA
        ema_val = feature_engine.ema(series, 10)
        # Both should be close to price for trending data
        assert abs(result.iloc[-1] - series.iloc[-1]) < abs(
            ema_val.iloc[-1] - series.iloc[-1]
        ) + 5

    def test_tema(self, feature_engine):
        series = pd.Series(np.linspace(1, 50, 50), dtype=float)
        result = feature_engine.tema(series, 10)
        assert len(result) == 50

    def test_kama_flat_market(self, feature_engine):
        # In flat market, KAMA should barely move
        series = pd.Series([100.0] * 50)
        result = feature_engine.kama(series, 10)
        valid = result.dropna()
        if len(valid) > 0:
            assert all(abs(v - 100) < 0.1 for v in valid)

    def test_kama_trending(self, feature_engine):
        series = pd.Series(np.linspace(100, 200, 50), dtype=float)
        result = feature_engine.kama(series, 10)
        valid = result.dropna()
        assert len(valid) > 0
        # Should trend upward
        assert valid.iloc[-1] > valid.iloc[0]


class TestFeatureEngineMomentum:
    def test_rsi_range(self, feature_engine):
        series = pd.Series(np.random.default_rng(42).normal(100, 2, 100).cumsum())
        result = feature_engine.rsi(series, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_overbought(self, feature_engine):
        # Strongly uptrending with small noise should yield high RSI
        rng = np.random.default_rng(42)
        base = np.linspace(100, 200, 50)
        series = pd.Series(base + rng.uniform(0, 0.5, 50), dtype=float)
        result = feature_engine.rsi(series, 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 70

    def test_rsi_oversold(self, feature_engine):
        # Strongly downtrending should yield low RSI
        series = pd.Series(np.linspace(200, 100, 50), dtype=float)
        result = feature_engine.rsi(series, 14)
        assert result.iloc[-1] < 30

    def test_stochastic_range(self, feature_engine, sample_ohlcv):
        k, d = feature_engine.stochastic(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        valid_k = k.dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()

    def test_macd_components(self, feature_engine):
        series = pd.Series(np.random.default_rng(42).normal(0, 1, 100).cumsum() + 100)
        line, signal, hist = feature_engine.macd(series)
        assert len(line) == 100
        assert len(signal) == 100
        # Histogram = line - signal
        valid_idx = line.dropna().index.intersection(signal.dropna().index)
        np.testing.assert_array_almost_equal(
            hist.loc[valid_idx].values,
            (line.loc[valid_idx] - signal.loc[valid_idx]).values,
            decimal=10,
        )

    def test_williams_r_range(self, feature_engine, sample_ohlcv):
        result = feature_engine.williams_r(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        valid = result.dropna()
        assert (valid >= -100).all()
        assert (valid <= 0).all()

    def test_cci(self, feature_engine, sample_ohlcv):
        result = feature_engine.cci(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        assert len(result) == len(sample_ohlcv)
        valid = result.dropna()
        assert len(valid) > 0

    def test_roc(self, feature_engine):
        series = pd.Series([100, 110, 121, 133], dtype=float)
        result = feature_engine.roc(series, 1)
        assert abs(result.iloc[1] - 10.0) < 0.01
        assert abs(result.iloc[2] - 10.0) < 0.01

    def test_mfi_range(self, feature_engine, sample_ohlcv):
        result = feature_engine.mfi(
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            sample_ohlcv["Close"],
            sample_ohlcv["Volume"],
        )
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestFeatureEngineVolatility:
    def test_atr_positive(self, feature_engine, sample_ohlcv):
        result = feature_engine.atr(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        valid = result.dropna()
        assert (valid > 0).all()

    def test_bollinger_bands_ordering(self, feature_engine):
        series = pd.Series(np.random.default_rng(42).normal(100, 5, 50))
        upper, mid, lower = feature_engine.bollinger_bands(series)
        valid_idx = upper.dropna().index.intersection(lower.dropna().index)
        assert (upper.loc[valid_idx] >= mid.loc[valid_idx]).all()
        assert (mid.loc[valid_idx] >= lower.loc[valid_idx]).all()

    def test_keltner_channels_ordering(self, feature_engine, sample_ohlcv):
        upper, mid, lower = feature_engine.keltner_channels(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        valid_idx = upper.dropna().index.intersection(lower.dropna().index)
        assert (upper.loc[valid_idx] >= lower.loc[valid_idx]).all()

    def test_donchian_channels(self, feature_engine, sample_ohlcv):
        upper, mid, lower = feature_engine.donchian_channels(
            sample_ohlcv["High"], sample_ohlcv["Low"]
        )
        valid_idx = upper.dropna().index.intersection(lower.dropna().index)
        assert (upper.loc[valid_idx] >= lower.loc[valid_idx]).all()
        # Middle should be between upper and lower
        np.testing.assert_array_almost_equal(
            mid.loc[valid_idx].values,
            ((upper.loc[valid_idx] + lower.loc[valid_idx]) / 2).values,
        )


class TestFeatureEngineVolume:
    def test_obv_direction(self, feature_engine):
        close = pd.Series([100, 101, 102, 103, 104], dtype=float)
        volume = pd.Series([1000, 1000, 1000, 1000, 1000])
        result = feature_engine.obv(close, volume)
        # All up days, OBV should increase
        assert result.iloc[-1] > result.iloc[1]

    def test_vwap(self, feature_engine, sample_ohlcv):
        result = feature_engine.vwap(
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            sample_ohlcv["Close"],
            sample_ohlcv["Volume"],
        )
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_cmf_range(self, feature_engine, sample_ohlcv):
        result = feature_engine.cmf(
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            sample_ohlcv["Close"],
            sample_ohlcv["Volume"],
        )
        valid = result.dropna()
        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_rvol(self, feature_engine):
        # 20 periods of 1000 then spike to 5000
        # Rolling mean at last point includes the spike: (19*1000+5000)/20=1200
        # So RVOL = 5000/1200 ≈ 4.17
        volume = pd.Series([1000] * 20 + [5000], dtype=float)
        result = feature_engine.rvol(volume, 20)
        assert result.iloc[-1] > 3.0  # Significantly above average

    def test_ad_line(self, feature_engine, sample_ohlcv):
        result = feature_engine.ad_line(
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            sample_ohlcv["Close"],
            sample_ohlcv["Volume"],
        )
        assert len(result) == len(sample_ohlcv)


class TestFeatureEngineTrendStrength:
    def test_adx_range(self, feature_engine, sample_ohlcv):
        result = feature_engine.adx(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        valid = result.dropna()
        # ADX is theoretically 0-100
        assert (valid >= 0).all()

    def test_ichimoku_components(self, feature_engine, sample_ohlcv):
        result = feature_engine.ichimoku(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        assert "tenkan" in result
        assert "kijun" in result
        assert "senkou_a" in result
        assert "senkou_b" in result
        assert "chikou" in result


class TestFeatureEngineStatistical:
    def test_hurst_exponent_range(self, feature_engine):
        series = pd.Series(np.random.default_rng(42).normal(0, 1, 500).cumsum())
        result = feature_engine.hurst_exponent(series)
        assert 0 <= result <= 1

    def test_hurst_short_series(self, feature_engine):
        series = pd.Series([1, 2, 3])
        result = feature_engine.hurst_exponent(series)
        assert result == 0.5  # Default for insufficient data

    def test_kalman_filter_smoothing(self, feature_engine):
        rng = np.random.default_rng(42)
        noisy = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 100)) + rng.normal(0, 0.3, 100))
        filtered = feature_engine.kalman_filter(noisy)
        # Filtered should be smoother (lower variance)
        assert filtered.std() < noisy.std()

    def test_kalman_filter_length(self, feature_engine):
        series = pd.Series(range(50), dtype=float)
        result = feature_engine.kalman_filter(series)
        assert len(result) == 50

    def test_fourier_transform_length(self, feature_engine):
        series = pd.Series(np.sin(np.linspace(0, 10, 100)))
        result = feature_engine.fourier_transform(series)
        assert len(result) == 100

    def test_approximate_entropy_positive(self, feature_engine):
        series = pd.Series(np.random.default_rng(42).normal(0, 1, 200))
        result = feature_engine.approximate_entropy(series)
        assert result >= 0

    def test_approximate_entropy_short_series(self, feature_engine):
        series = pd.Series([1, 2])
        result = feature_engine.approximate_entropy(series)
        assert result == 0.0

    def test_fractal_dimension_range(self, feature_engine):
        series = pd.Series(np.random.default_rng(42).normal(0, 1, 300).cumsum())
        result = feature_engine.fractal_dimension(series)
        assert 1 <= result <= 2

    def test_shannon_entropy_positive(self, feature_engine):
        series = pd.Series(np.random.default_rng(42).normal(100, 5, 200))
        result = feature_engine.shannon_entropy(series)
        assert result > 0

    def test_shannon_entropy_constant(self, feature_engine):
        series = pd.Series([100.0] * 50)
        result = feature_engine.shannon_entropy(series)
        # Constant series has no variation in returns
        assert result == 0.0

    def test_kyle_lambda(self, feature_engine):
        close = pd.Series(np.random.default_rng(42).normal(100, 2, 50).cumsum())
        volume = pd.Series(np.random.default_rng(43).integers(1000, 10000, 50))
        result = feature_engine.kyle_lambda(close, volume, period=10)
        assert len(result) == 50

    def test_vpin_range(self, feature_engine):
        close = pd.Series(np.random.default_rng(42).normal(0, 1, 100).cumsum() + 100)
        volume = pd.Series(np.random.default_rng(43).integers(1000, 10000, 100))
        result = feature_engine.vpin(close, volume, n_buckets=20)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.5).all()  # Theoretically 0-1 but can exceed slightly


class TestFeatureEnginePatterns:
    def test_harmonic_patterns_structure(self, feature_engine, sample_ohlcv):
        result = feature_engine.detect_harmonic_patterns(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        )
        assert "gartley" in result
        assert "butterfly" in result
        assert "crab" in result
        assert "bat" in result

    def test_candlestick_patterns(self, feature_engine, sample_ohlcv):
        result = feature_engine.detect_candlestick_patterns(
            sample_ohlcv["Open"],
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            sample_ohlcv["Close"],
        )
        assert "doji" in result.columns
        assert "hammer" in result.columns
        assert "bullish_engulfing" in result.columns
        assert "bearish_engulfing" in result.columns
        assert "morning_star" in result.columns
        # All values should be 0 or 1
        assert set(result.values.flatten()).issubset({0, 1})

    def test_swing_points(self, feature_engine):
        high = pd.Series([1, 3, 2, 1, 5, 3, 2, 1, 4, 2, 1], dtype=float)
        low = pd.Series([0, 2, 1, 0, 4, 2, 1, 0, 3, 1, 0], dtype=float)
        result = feature_engine._find_swing_points(high, low, window=2)
        assert len(result) > 0


class TestFeatureEngineMultiTimeframe:
    def test_resample_weekly(self, feature_engine, sample_ohlcv):
        weekly = feature_engine.resample_to_weekly(sample_ohlcv)
        assert len(weekly) < len(sample_ohlcv)
        assert len(weekly) > 0

    def test_resample_monthly(self, feature_engine, sample_ohlcv):
        monthly = feature_engine.resample_to_monthly(sample_ohlcv)
        assert len(monthly) < len(sample_ohlcv)
        assert len(monthly) > 0

    def test_compute_all_features(self, feature_engine, sample_ohlcv):
        result = feature_engine.compute_all_features(sample_ohlcv)
        assert len(result.columns) >= 50
        assert len(result) == len(sample_ohlcv)

    def test_compute_all_features_empty(self, feature_engine):
        df = pd.DataFrame()
        result = feature_engine.compute_all_features(df)
        assert result.empty


# ============================================================================
# REGIME DETECTOR TESTS
# ============================================================================


class TestRegimeDetector:
    def test_fit_basic(self):
        detector = RegimeDetector(n_states=3)
        returns = pd.Series(np.random.default_rng(42).normal(0, 0.02, 500))
        detector.fit(returns)
        assert detector.emission_means is not None
        assert detector.emission_stds is not None
        assert detector.transition_matrix is not None

    def test_fit_insufficient_data(self):
        detector = RegimeDetector(n_states=3)
        returns = pd.Series([0.01, 0.02, 0.03])
        detector.fit(returns)
        # Should handle gracefully without crashing

    def test_predict_regime_output(self):
        detector = RegimeDetector(n_states=3)
        returns = pd.Series(
            np.random.default_rng(42).normal(0, 0.02, 200),
            index=pd.date_range("2024-01-01", periods=200, freq="B"),
        )
        result = detector.predict_regime(returns)
        assert len(result) > 0
        assert all(isinstance(r, RegimeState) for r in result)

    def test_transition_matrix_stochastic(self):
        detector = RegimeDetector(n_states=3)
        returns = pd.Series(np.random.default_rng(42).normal(0, 0.02, 300))
        detector.fit(returns)
        # Each row should sum to ~1
        row_sums = detector.transition_matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=5)

    def test_crash_risk_filter(self):
        detector = RegimeDetector()
        # Normal period followed by high-vol crash
        returns = pd.Series(
            np.concatenate([
                np.random.default_rng(42).normal(0, 0.01, 252),
                np.random.default_rng(42).normal(-0.02, 0.05, 50),
            ]),
            index=pd.date_range("2023-01-01", periods=302, freq="B"),
        )
        crash_flags = detector.crash_risk_filter(returns)
        # Some of the volatile period should be flagged
        assert crash_flags.sum() > 0

    def test_state_mapping(self):
        detector = RegimeDetector(n_states=3)
        detector.emission_means = np.array([-0.01, 0.001, 0.02])
        mapping = detector._map_states_to_regimes()
        assert mapping[0] == RegimeState.BEAR_DISTRIBUTION
        assert mapping[2] == RegimeState.BULL_ACCUMULATION


# ============================================================================
# DYNAMIC ADAPTIVE ENSEMBLE TESTS
# ============================================================================


class TestDynamicAdaptiveEnsemble:
    def test_fit_basic(self):
        ensemble = DynamicAdaptiveEnsemble()
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame(rng.normal(0, 1, (n, 10)))
        y = pd.Series(X.iloc[:, 0] * 0.5 + rng.normal(0, 0.1, n))
        results = ensemble.fit(X, y)
        assert len(results) > 0
        assert ensemble._is_fitted

    def test_fit_insufficient_data(self):
        ensemble = DynamicAdaptiveEnsemble()
        X = pd.DataFrame(np.ones((5, 3)))
        y = pd.Series([1, 2, 3, 4, 5])
        results = ensemble.fit(X, y)
        assert results == {}

    def test_predict_shape(self):
        ensemble = DynamicAdaptiveEnsemble()
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame(rng.normal(0, 1, (n, 10)))
        y = pd.Series(X.iloc[:, 0] * 0.5 + rng.normal(0, 0.1, n))
        ensemble.fit(X, y)

        X_test = pd.DataFrame(rng.normal(0, 1, (20, 10)))
        preds, confs = ensemble.predict(X_test)
        assert len(preds) == 20
        assert len(confs) == 20

    def test_predict_without_fit(self):
        ensemble = DynamicAdaptiveEnsemble()
        X = pd.DataFrame(np.ones((5, 3)))
        preds, confs = ensemble.predict(X)
        assert (preds == 0).all()
        assert (confs == 0).all()

    def test_weights_sum_to_one(self):
        ensemble = DynamicAdaptiveEnsemble()
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame(rng.normal(0, 1, (n, 10)))
        y = pd.Series(X.iloc[:, 0] * 0.5 + rng.normal(0, 0.1, n))
        ensemble.fit(X, y)
        assert abs(sum(ensemble.weights.values()) - 1.0) < 0.01

    def test_linear_model_fit(self):
        ensemble = DynamicAdaptiveEnsemble()
        X = np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
        y = np.array([1, 1, 2, 3], dtype=float)
        model = ensemble._fit_linear(X, y)
        assert model["type"] == "linear"
        assert len(model["coeffs"]) == 2

    def test_gradient_boost_fit(self):
        ensemble = DynamicAdaptiveEnsemble()
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 5))
        y = X[:, 0] + rng.normal(0, 0.1, 100)
        model = ensemble._fit_gradient_boost(X, y, n_trees=10)
        assert model["type"] == "gradient_boost"
        assert len(model["trees"]) == 10

    def test_random_forest_fit(self):
        ensemble = DynamicAdaptiveEnsemble()
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 5))
        y = X[:, 0] + rng.normal(0, 0.1, 100)
        model = ensemble._fit_random_forest(X, y, n_trees=10)
        assert model["type"] == "random_forest"
        assert len(model["trees"]) == 10


# ============================================================================
# SIGNAL GENERATOR TESTS
# ============================================================================


class TestSignalGenerator:
    def test_trend_score_range(self, sample_ohlcv):
        fe = FeatureEngine()
        features = fe.compute_all_features(sample_ohlcv)
        sg = SignalGenerator()
        score = sg.compute_trend_score(features)
        valid = score.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_institutional_score_range(self, sample_ohlcv):
        fe = FeatureEngine()
        features = fe.compute_all_features(sample_ohlcv)
        sg = SignalGenerator()
        score = sg.compute_institutional_score(features)
        valid = score.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_mean_reversion_score_range(self, sample_ohlcv):
        fe = FeatureEngine()
        features = fe.compute_all_features(sample_ohlcv)
        sg = SignalGenerator()
        score = sg.compute_mean_reversion_score(features)
        valid = score.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_microstructure_score_range(self, sample_ohlcv):
        fe = FeatureEngine()
        features = fe.compute_all_features(sample_ohlcv)
        sg = SignalGenerator()
        score = sg.compute_microstructure_score(features)
        valid = score.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_generate_signals_output(self, sample_ohlcv):
        fe = FeatureEngine()
        features = fe.compute_all_features(sample_ohlcv)
        sg = SignalGenerator()
        regime = pd.Series(
            RegimeState.BULL_ACCUMULATION, index=features.index
        )
        preds = np.random.default_rng(42).uniform(0, 1, len(features))
        confs = np.random.default_rng(43).uniform(0.3, 0.9, len(features))
        result = sg.generate_signals(features, regime, preds, confs)
        assert "signal" in result.columns
        assert "confidence" in result.columns
        assert "composite_score" in result.columns
        valid_signals = {"BUY", "SELL", "HOLD", "STAY_AWAY"}
        assert set(result["signal"].unique()).issubset(valid_signals)

    def test_crash_regime_overrides_to_stay_away(self, sample_ohlcv):
        fe = FeatureEngine()
        features = fe.compute_all_features(sample_ohlcv)
        sg = SignalGenerator()
        regime = pd.Series(RegimeState.CRASH_RISK, index=features.index)
        preds = np.ones(len(features))  # Strong buy prediction
        confs = np.ones(len(features))
        result = sg.generate_signals(features, regime, preds, confs)
        assert (result["signal"] == "STAY_AWAY").all()

    def test_decision_weights_sum(self):
        sg = SignalGenerator()
        total = (
            sg.TREND_WEIGHT
            + sg.INSTITUTIONAL_WEIGHT
            + sg.MEAN_REVERSION_WEIGHT
            + sg.MICROSTRUCTURE_WEIGHT
        )
        assert abs(total - 1.0) < 0.001


# ============================================================================
# RISK ENGINE TESTS
# ============================================================================


class TestRiskEngine:
    def test_transaction_costs_positive(self, risk_engine):
        costs = risk_engine.calculate_transaction_costs(1000.0, 100, "BUY")
        assert costs.total > 0
        assert costs.brokerage > 0
        assert costs.stt > 0
        assert costs.gst > 0
        assert costs.slippage > 0

    def test_transaction_costs_sell_vs_buy(self, risk_engine):
        buy_costs = risk_engine.calculate_transaction_costs(1000.0, 100, "BUY")
        sell_costs = risk_engine.calculate_transaction_costs(1000.0, 100, "SELL")
        # Sell has CDSL, Buy has stamp duty
        assert buy_costs.stamp_duty > 0
        assert sell_costs.stamp_duty == 0
        assert sell_costs.cdsl > 0
        assert buy_costs.cdsl == 0

    def test_fractional_kelly_range(self, risk_engine):
        kelly = risk_engine.fractional_kelly(0.6, 0.03, 0.015)
        assert KELLY_FRACTION_MIN <= kelly <= KELLY_FRACTION_MAX

    def test_fractional_kelly_zero_loss(self, risk_engine):
        kelly = risk_engine.fractional_kelly(0.6, 0.03, 0.0)
        assert kelly == KELLY_FRACTION_MIN

    def test_fractional_kelly_low_win_rate(self, risk_engine):
        kelly = risk_engine.fractional_kelly(0.3, 0.02, 0.03)
        # Low win rate with unfavorable odds should give min kelly
        assert kelly == KELLY_FRACTION_MIN

    def test_position_size_respects_risk_rule(self, risk_engine):
        entry = 1000.0
        stop = 980.0  # 20 risk per share
        size = risk_engine.position_size(entry, stop)
        max_risk = risk_engine.current_capital * RISK_PER_TRADE
        actual_risk = size * 20
        assert actual_risk <= max_risk * 1.01  # 1% tolerance

    def test_position_size_zero_risk(self, risk_engine):
        size = risk_engine.position_size(1000.0, 1000.0)  # No risk
        assert size == 0

    def test_position_size_portfolio_heat_limit(self, risk_engine):
        risk_engine.portfolio_heat = 0.075  # Near limit
        size = risk_engine.position_size(1000.0, 980.0)
        # Should be reduced due to heat limit
        unrestricted = RiskEngine(base_capital=300_000).position_size(1000.0, 980.0)
        assert size <= unrestricted

    def test_stop_loss_below_entry(self, risk_engine):
        stop = risk_engine.calculate_stop_loss(100.0, 3.0, 2.0)
        assert stop < 100.0
        assert stop == 94.0

    def test_trailing_stop(self, risk_engine):
        stop = risk_engine.calculate_trailing_stop(110.0, 115.0, 3.0)
        # Based on highest price, not current
        assert stop == 115.0 - 3.0 * 2.5

    def test_target_above_entry(self, risk_engine):
        target = risk_engine.calculate_target(100.0, 95.0, 2.5)
        assert target > 100.0
        assert target == 100.0 + (5.0 * 2.5)

    def test_circuit_breaker_below_limit(self, risk_engine):
        risk_engine.portfolio_heat = 0.05
        assert not risk_engine.check_circuit_breaker()

    def test_circuit_breaker_at_limit(self, risk_engine):
        risk_engine.portfolio_heat = PORTFOLIO_HEAT_LIMIT
        assert risk_engine.check_circuit_breaker()

    def test_circuit_breaker_above_limit(self, risk_engine):
        risk_engine.portfolio_heat = 0.10
        assert risk_engine.check_circuit_breaker()

    def test_var_calculation(self, risk_engine):
        returns = pd.Series(np.random.default_rng(42).normal(0, 0.02, 252))
        var = risk_engine.calculate_var(returns)
        assert var > 0

    def test_var_empty_returns(self, risk_engine):
        returns = pd.Series(dtype=float)
        var = risk_engine.calculate_var(returns)
        assert var == 0.0

    def test_update_portfolio_heat(self, risk_engine):
        risk_engine.positions = {
            "A": {"risk_amount": 1500},
            "B": {"risk_amount": 1000},
        }
        risk_engine.update_portfolio_heat()
        expected = 2500 / 300_000
        assert abs(risk_engine.portfolio_heat - expected) < 0.0001


# ============================================================================
# POSITION RECONCILER TESTS
# ============================================================================


class TestPositionReconciler:
    def test_record_entry(self):
        reconciler = PositionReconciler()
        costs = TransactionCost(brokerage=20, stt=100, gst=3.6)
        entry = reconciler.record_entry("TEST.NS", 500.0, 100, costs)
        assert entry.symbol == "TEST.NS"
        assert entry.entry_price == 500.0
        assert entry.quantity == 100
        assert len(reconciler.ledger) == 1

    def test_record_exit_profitable(self):
        reconciler = PositionReconciler()
        entry_costs = TransactionCost(brokerage=20, stt=50)
        reconciler.record_entry("TEST.NS", 100.0, 50, entry_costs)
        exit_costs = TransactionCost(brokerage=20, stt=55)
        result = reconciler.record_exit("TEST.NS", 110.0, exit_costs)
        assert result is not None
        assert result.pnl > 0  # 50 shares * 10 = 500 - costs
        assert result.exit_price == 110.0

    def test_record_exit_loss(self):
        reconciler = PositionReconciler()
        entry_costs = TransactionCost(brokerage=20)
        reconciler.record_entry("LOSS.NS", 100.0, 50, entry_costs)
        exit_costs = TransactionCost(brokerage=20)
        result = reconciler.record_exit("LOSS.NS", 90.0, exit_costs)
        assert result is not None
        assert result.pnl < 0

    def test_record_exit_nonexistent_symbol(self):
        reconciler = PositionReconciler()
        exit_costs = TransactionCost()
        result = reconciler.record_exit("FAKE.NS", 100.0, exit_costs)
        assert result is None

    def test_xirr_calculation(self):
        reconciler = PositionReconciler()
        entry_costs = TransactionCost(brokerage=20)
        reconciler.record_entry("XIRR.NS", 100.0, 100, entry_costs)
        # Simulate passage of time
        reconciler.ledger[-1].entry_date = datetime(2024, 1, 1)
        exit_costs = TransactionCost(brokerage=20)
        result = reconciler.record_exit("XIRR.NS", 120.0, exit_costs)
        # Force exit date for deterministic XIRR
        result.exit_date = datetime(2024, 7, 1)
        xirr = reconciler._compute_xirr(result)
        assert xirr > 0  # 20% gain in 6 months

    def test_performance_summary_empty(self):
        reconciler = PositionReconciler()
        summary = reconciler.get_performance_summary()
        assert summary["total_trades"] == 0

    def test_performance_summary_mixed(self):
        reconciler = PositionReconciler()
        # Win
        reconciler.record_entry("W.NS", 100.0, 10, TransactionCost())
        reconciler.record_exit("W.NS", 110.0, TransactionCost())
        # Loss
        reconciler.record_entry("L.NS", 100.0, 10, TransactionCost())
        reconciler.record_exit("L.NS", 95.0, TransactionCost())

        summary = reconciler.get_performance_summary()
        assert summary["total_trades"] == 2
        assert summary["winning_trades"] == 1
        assert summary["losing_trades"] == 1
        assert summary["win_rate"] == 0.5

    def test_max_drawdown(self):
        reconciler = PositionReconciler()
        pnls = [100, 50, -200, 100, -50]
        dd = reconciler._max_drawdown(pnls)
        assert dd < 0

    def test_sharpe_ratio(self):
        reconciler = PositionReconciler()
        pnls = [100, 100, 100, 100, 100]  # Consistent profits
        sharpe = reconciler._sharpe_ratio(pnls)
        # With constant returns, std = 0, sharpe = 0
        assert sharpe == 0.0

    def test_sharpe_ratio_varied(self):
        reconciler = PositionReconciler()
        rng = np.random.default_rng(42)
        pnls = (rng.normal(50, 20, 50)).tolist()
        sharpe = reconciler._sharpe_ratio(pnls)
        # Should be computable
        assert isinstance(sharpe, float)


# ============================================================================
# SYSTEM MONITOR TESTS
# ============================================================================


class TestSystemMonitor:
    def test_get_system_info(self):
        monitor = SystemMonitor()
        info = monitor.get_system_info()
        assert "cpu_count" in info
        assert info["cpu_count"] >= 1

    def test_recommend_batch_size(self):
        monitor = SystemMonitor()
        batch = monitor.recommend_batch_size()
        assert batch > 0
        assert batch <= 100

    def test_graceful_degradation(self):
        monitor = SystemMonitor()
        result = monitor.graceful_degradation("test_component", ValueError("test"))
        assert "degraded" in result
        assert "test_component" in result


# ============================================================================
# INTEGRATION TESTS - NSE500AlphaArchitect
# ============================================================================


class TestNSE500AlphaArchitect:
    def test_initialization(self):
        arch = NSE500AlphaArchitect(capital=500_000)
        assert arch.capital == 500_000
        assert arch.data_manager is not None
        assert arch.feature_engine is not None
        assert arch.risk_engine.base_capital == 500_000

    def test_initialize_loads_universe(self):
        arch = NSE500AlphaArchitect()
        info = arch.initialize()
        assert info["universe_size"] > 0
        assert "system_info" in info
        assert "batch_size" in info

    def test_data_pipeline(self):
        arch = NSE500AlphaArchitect()
        arch.initialize()
        results = arch.run_data_pipeline(symbols=arch.universe[:3])
        assert results["fetched"] + results["cached"] == 3
        assert results["failed"] == 0
        assert len(arch.data_cache) == 3

    def test_feature_engineering(self):
        arch = NSE500AlphaArchitect()
        arch.initialize()
        arch.run_data_pipeline(symbols=arch.universe[:2])
        results = arch.run_feature_engineering()
        assert results["computed"] == 2
        assert len(arch.feature_cache) == 2

    def test_regime_detection(self):
        arch = NSE500AlphaArchitect()
        arch.initialize()
        arch.run_data_pipeline(symbols=arch.universe[:2])
        regimes = arch.run_regime_detection()
        assert len(regimes) > 0

    def test_full_pipeline(self):
        arch = NSE500AlphaArchitect(capital=300_000)
        summary = arch.run_full_pipeline()
        assert summary["elapsed_seconds"] > 0
        assert summary["log_integrity"]
        assert "signals_generated" in summary

    def test_get_top_signals(self):
        arch = NSE500AlphaArchitect()
        arch.run_full_pipeline()
        top = arch.get_top_signals(5)
        assert isinstance(top, list)
        for s in top:
            assert "symbol" in s
            assert "signal" in s
            assert "confidence" in s

    def test_custom_capital(self):
        arch = NSE500AlphaArchitect(capital=1_000_000)
        assert arch.risk_engine.base_capital == 1_000_000
        assert arch.risk_engine.current_capital == 1_000_000


# ============================================================================
# DATA CLASSES TESTS
# ============================================================================


class TestDataClasses:
    def test_trade_signal_to_dict(self):
        signal = TradeSignal(
            symbol="TEST.NS",
            signal=Signal.BUY,
            confidence=0.85,
            predicted_return=0.032,
            regime=RegimeState.BULL_ACCUMULATION,
            entry_price=500.0,
            stop_loss=485.0,
            target_price=537.5,
            position_size=200,
            risk_amount=3000.0,
        )
        d = signal.to_dict()
        assert d["symbol"] == "TEST.NS"
        assert d["signal"] == "BUY"
        assert d["confidence"] == 0.85
        assert d["regime"] == "BULL_ACCUMULATION"

    def test_transaction_cost_total(self):
        cost = TransactionCost(
            brokerage=20, stt=100, gst=3.6, sebi=0.1, stamp_duty=15, cdsl=13.5, slippage=50
        )
        expected = 20 + 100 + 3.6 + 0.1 + 15 + 13.5 + 50
        assert abs(cost.total - expected) < 0.01

    def test_transaction_cost_zero(self):
        cost = TransactionCost()
        assert cost.total == 0.0

    def test_signal_enum(self):
        assert Signal.BUY.name == "BUY"
        assert Signal.SELL.name == "SELL"
        assert Signal.HOLD.name == "HOLD"
        assert Signal.STAY_AWAY.name == "STAY_AWAY"

    def test_regime_state_enum(self):
        assert RegimeState.BULL_ACCUMULATION.name == "BULL_ACCUMULATION"
        assert RegimeState.BEAR_DISTRIBUTION.name == "BEAR_DISTRIBUTION"
        assert RegimeState.NEUTRAL_CHOP.name == "NEUTRAL_CHOP"
        assert RegimeState.CRASH_RISK.name == "CRASH_RISK"


# ============================================================================
# EDGE CASES & ROBUSTNESS
# ============================================================================


class TestEdgeCases:
    def test_feature_engine_single_row(self, feature_engine):
        df = pd.DataFrame(
            {"Open": [100], "High": [105], "Low": [95], "Close": [102], "Volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1, freq="B"),
        )
        result = feature_engine.compute_all_features(df)
        assert len(result) == 1

    def test_risk_engine_very_small_capital(self):
        engine = RiskEngine(base_capital=1000)
        size = engine.position_size(500.0, 490.0)
        # 1% of 1000 = 10, risk per share = 10, so max 1 share
        assert size >= 0
        assert size <= 2

    def test_risk_engine_very_large_capital(self):
        engine = RiskEngine(base_capital=100_000_000)
        size = engine.position_size(1000.0, 990.0)
        assert size > 0

    def test_ensemble_all_nan_features(self):
        ensemble = DynamicAdaptiveEnsemble()
        X = pd.DataFrame(np.nan, index=range(100), columns=range(5))
        y = pd.Series(range(100), dtype=float)
        ensemble.fit(X, y)
        # Should handle gracefully

    def test_regime_detector_constant_returns(self):
        detector = RegimeDetector(n_states=3)
        returns = pd.Series(
            np.zeros(200),
            index=pd.date_range("2024-01-01", periods=200, freq="B"),
        )
        # Should not crash
        detector.predict_regime(returns)

    def test_feature_engine_all_same_price(self, feature_engine):
        n = 50
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [100.0] * n,
                "Low": [100.0] * n,
                "Close": [100.0] * n,
                "Volume": [1000] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
        )
        # Should not crash even with zero variance
        result = feature_engine.compute_all_features(df)
        assert len(result) == n

    def test_position_reconciler_multiple_entries_same_symbol(self):
        reconciler = PositionReconciler()
        reconciler.record_entry("MULTI.NS", 100.0, 50, TransactionCost())
        reconciler.record_entry("MULTI.NS", 105.0, 30, TransactionCost())
        # Exit should close most recent
        result = reconciler.record_exit("MULTI.NS", 110.0, TransactionCost())
        assert result is not None
        assert result.entry_price == 105.0  # Most recent entry

    def test_data_manager_comments_in_universe(self, tmp_path):
        universe_file = tmp_path / "NSE500.txt"
        universe_file.write_text("# Comment\nRELIANCE.NS\n\n# Another\nTCS.NS\n")
        dm = NSE500DataManager(
            universe_file=str(universe_file),
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
        )
        symbols = dm.load_universe()
        assert symbols == ["RELIANCE.NS", "TCS.NS"]


# ============================================================================
# STATE MANAGER TESTS
# ============================================================================


class TestStateManager:
    def test_save_and_load(self, tmp_path):
        sm = StateManager(state_dir=str(tmp_path / "state"))
        assert sm.save_state("test", {"key": "value", "num": 42})
        loaded = sm.load_state("test")
        assert loaded["key"] == "value"
        assert loaded["num"] == 42

    def test_load_missing(self, tmp_path):
        sm = StateManager(state_dir=str(tmp_path / "state"))
        assert sm.load_state("nonexistent") == {}

    def test_delete_state(self, tmp_path):
        sm = StateManager(state_dir=str(tmp_path / "state"))
        sm.save_state("todel", {"x": 1})
        assert sm.delete_state("todel")
        assert sm.load_state("todel") == {}

    def test_list_states(self, tmp_path):
        sm = StateManager(state_dir=str(tmp_path / "state"))
        sm.save_state("alpha", {"a": 1})
        sm.save_state("beta", {"b": 2})
        states = sm.list_states()
        assert "alpha" in states
        assert "beta" in states

    def test_atomic_overwrite(self, tmp_path):
        sm = StateManager(state_dir=str(tmp_path / "state"))
        sm.save_state("x", {"v": 1})
        sm.save_state("x", {"v": 2})
        assert sm.load_state("x")["v"] == 2

    def test_datetime_serialization(self, tmp_path):
        sm = StateManager(state_dir=str(tmp_path / "state"))
        assert sm.save_state("dt", {"ts": datetime.now()})
        loaded = sm.load_state("dt")
        assert "ts" in loaded


# ============================================================================
# STRESS TESTING FRAMEWORK TESTS
# ============================================================================


class TestStressTestingFramework:
    @pytest.fixture
    def stress_tester(self):
        return StressTestingFramework(n_simulations=500, seed=42)

    @pytest.fixture
    def sample_returns(self):
        return pd.Series(np.random.default_rng(42).normal(0.001, 0.02, 252))

    def test_monte_carlo_var_positive(self, stress_tester, sample_returns):
        result = stress_tester.monte_carlo_var(sample_returns, portfolio_value=300_000)
        assert result["var"] > 0
        assert result["cvar"] > 0
        assert result["max_loss"] > 0

    def test_cvar_exceeds_var(self, stress_tester, sample_returns):
        result = stress_tester.monte_carlo_var(sample_returns)
        assert result["cvar"] >= result["var"]

    def test_scenario_analysis_count(self, stress_tester, sample_returns):
        scenarios = stress_tester.scenario_analysis(sample_returns)
        assert len(scenarios) == 6

    def test_scenario_analysis_keys(self, stress_tester, sample_returns):
        scenarios = stress_tester.scenario_analysis(sample_returns)
        for s in scenarios:
            assert "scenario" in s
            assert "expected_loss" in s
            assert "worst_case" in s
            assert "recovery_probability" in s

    def test_insufficient_data(self, stress_tester):
        short = pd.Series([0.01, 0.02, -0.01])
        result = stress_tester.monte_carlo_var(short)
        assert result["var"] == 0.0

    def test_scenario_insufficient_data(self, stress_tester):
        short = pd.Series([0.01])
        assert stress_tester.scenario_analysis(short) == []

    def test_different_confidence_levels(self, stress_tester, sample_returns):
        var_95 = stress_tester.monte_carlo_var(sample_returns, confidence=0.95)
        var_99 = stress_tester.monte_carlo_var(sample_returns, confidence=0.99)
        assert var_99["var"] >= var_95["var"]


# ============================================================================
# WALK-FORWARD OPTIMIZER TESTS
# ============================================================================


class TestWalkForwardOptimizer:
    def test_generate_splits_basic(self):
        wfo = WalkForwardOptimizer(
            train_periods=100, val_periods=30, test_periods=20, step_size=20
        )
        splits = wfo.generate_splits(200)
        assert len(splits) >= 1

    def test_split_structure(self):
        wfo = WalkForwardOptimizer(
            train_periods=100, val_periods=30, test_periods=20, step_size=20
        )
        splits = wfo.generate_splits(200)
        for split in splits:
            assert "train" in split
            assert "val" in split
            assert "test" in split
            train_s, train_e = split["train"]
            val_s, val_e = split["val"]
            test_s, test_e = split["test"]
            assert train_e == val_s
            assert val_e == test_s
            assert train_e - train_s == 100
            assert val_e - val_s == 30

    def test_insufficient_samples(self):
        wfo = WalkForwardOptimizer(
            train_periods=100, val_periods=50, test_periods=30
        )
        splits = wfo.generate_splits(50)
        assert len(splits) == 0

    def test_rolling_advance(self):
        wfo = WalkForwardOptimizer(
            train_periods=50, val_periods=10, test_periods=10, step_size=10
        )
        splits = wfo.generate_splits(100)
        if len(splits) >= 2:
            assert splits[1]["train"][0] == splits[0]["train"][0] + 10


# ============================================================================
# CONCEPT DRIFT DETECTOR TESTS
# ============================================================================


class TestConceptDriftDetector:
    def test_set_baseline(self):
        dd = ConceptDriftDetector()
        df = pd.DataFrame({
            "a": np.random.default_rng(1).normal(0, 1, 100),
            "b": np.random.default_rng(2).normal(5, 2, 100),
        })
        dd.set_baseline(df)
        assert "a" in dd.baseline_distributions
        assert "b" in dd.baseline_distributions

    def test_no_drift_similar_data(self):
        dd = ConceptDriftDetector(threshold=0.5)
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        baseline = pd.DataFrame({"a": rng1.normal(0, 1, 200)})
        dd.set_baseline(baseline)
        current = pd.DataFrame({"a": rng2.normal(0, 1, 200)})
        result = dd.detect_drift(current)
        assert result["drifted"] is False

    def test_drift_shifted_data(self):
        dd = ConceptDriftDetector(threshold=0.05)
        baseline = pd.DataFrame({"a": np.random.default_rng(1).normal(0, 1, 200)})
        dd.set_baseline(baseline)
        shifted = pd.DataFrame({"a": np.random.default_rng(2).normal(10, 5, 200)})
        result = dd.detect_drift(shifted)
        assert result["n_drifted"] > 0

    def test_no_baseline(self):
        dd = ConceptDriftDetector()
        result = dd.detect_drift(pd.DataFrame({"a": [1, 2, 3]}))
        assert result["drifted"] is False

    def test_compute_psi_identical(self):
        dd = ConceptDriftDetector()
        dist = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        psi = dd.compute_psi(dist, dist)
        assert abs(psi) < 1e-10

    def test_compute_psi_different(self):
        dd = ConceptDriftDetector()
        base = np.array([0.5, 0.3, 0.2])
        curr = np.array([0.1, 0.3, 0.6])
        psi = dd.compute_psi(base, curr)
        assert psi > 0


# ============================================================================
# COINTEGRATION ANALYZER TESTS
# ============================================================================


class TestCointegrationAnalyzer:
    def test_cointegrated_series(self):
        ca = CointegrationAnalyzer()
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n).cumsum() + 100
        noise = rng.normal(0, 0.5, n)
        y = 2 * x + noise + 50
        result = ca.engle_granger_test(pd.Series(y), pd.Series(x))
        assert "hedge_ratio" in result
        assert result["half_life"] >= 0

    def test_non_cointegrated(self):
        ca = CointegrationAnalyzer()
        rng = np.random.default_rng(42)
        a = pd.Series(rng.normal(0, 1, 500).cumsum())
        b = pd.Series(rng.normal(0, 1, 500).cumsum())
        result = ca.engle_granger_test(a, b)
        assert "cointegrated" in result

    def test_short_series(self):
        ca = CointegrationAnalyzer()
        a = pd.Series([1, 2, 3])
        b = pd.Series([4, 5, 6])
        result = ca.engle_granger_test(a, b)
        assert result["cointegrated"] is False

    def test_find_pairs(self):
        ca = CointegrationAnalyzer()
        rng = np.random.default_rng(42)
        n = 500
        base = rng.normal(0, 1, n).cumsum()
        prices = {
            "A": pd.Series(base + rng.normal(0, 0.1, n)),
            "B": pd.Series(2 * base + rng.normal(0, 0.2, n) + 10),
            "C": pd.Series(rng.normal(0, 1, n).cumsum()),
        }
        pairs = ca.find_cointegrated_pairs(prices)
        assert isinstance(pairs, list)


# ============================================================================
# CORPORATE ACTIONS HANDLER TESTS
# ============================================================================


class TestCorporateActionsHandler:
    def test_detect_2_to_1_split(self):
        cah = CorporateActionsHandler()
        df = pd.DataFrame(
            {"Close": [1000.0, 1010.0, 500.0, 505.0]},
            index=pd.date_range("2024-01-01", periods=4, freq="B"),
        )
        splits = cah.detect_splits(df)
        assert len(splits) >= 1
        assert splits[0]["type"] == "split"

    def test_no_split_normal_price(self):
        cah = CorporateActionsHandler()
        df = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0, 103.0]},
            index=pd.date_range("2024-01-01", periods=4, freq="B"),
        )
        splits = cah.detect_splits(df)
        assert len(splits) == 0

    def test_adjust_for_splits(self):
        cah = CorporateActionsHandler()
        df = pd.DataFrame(
            {
                "Open": [990, 1005, 495, 500],
                "High": [1020, 1015, 510, 515],
                "Low": [980, 1000, 490, 498],
                "Close": [1000, 1010, 500, 505],
                "Volume": [10000, 12000, 25000, 22000],
            },
            index=pd.date_range("2024-01-01", periods=4, freq="B"),
        )
        splits = cah.detect_splits(df)
        adjusted = cah.adjust_for_splits(df, splits)
        assert len(adjusted) == 4

    def test_empty_dataframe(self):
        cah = CorporateActionsHandler()
        assert cah.detect_splits(pd.DataFrame()) == []

    def test_no_splits_returns_original(self):
        cah = CorporateActionsHandler()
        df = pd.DataFrame({"Close": [100, 101]}, index=[0, 1])
        result = cah.adjust_for_splits(df, [])
        assert result.equals(df)


# ============================================================================
# PERFORMANCE ANALYTICS TESTS
# ============================================================================


class TestPerformanceAnalytics:
    @pytest.fixture
    def pa(self):
        return PerformanceAnalytics()

    @pytest.fixture
    def sample_returns(self):
        return pd.Series(
            np.random.default_rng(42).normal(0.001, 0.015, 252),
            index=pd.date_range("2024-01-01", periods=252, freq="B"),
        )

    @pytest.fixture
    def bench_returns(self):
        return pd.Series(
            np.random.default_rng(99).normal(0.0005, 0.012, 252),
            index=pd.date_range("2024-01-01", periods=252, freq="B"),
        )

    def test_alpha_beta(self, pa, sample_returns, bench_returns):
        result = pa.compute_alpha_beta(sample_returns, bench_returns)
        assert "alpha" in result
        assert "beta" in result
        assert "r_squared" in result

    def test_sortino_ratio(self, pa, sample_returns):
        sortino = pa.sortino_ratio(sample_returns)
        assert isinstance(sortino, float)

    def test_calmar_ratio(self, pa, sample_returns):
        calmar = pa.calmar_ratio(sample_returns)
        assert isinstance(calmar, float)

    def test_information_ratio(self, pa, sample_returns, bench_returns):
        ir = pa.information_ratio(sample_returns, bench_returns)
        assert isinstance(ir, float)

    def test_full_attribution(self, pa, sample_returns, bench_returns):
        result = pa.full_attribution(sample_returns, bench_returns)
        assert "total_return" in result
        assert "annualized_return" in result
        assert "volatility" in result
        assert "max_drawdown" in result
        assert "alpha" in result
        assert "sortino" in result

    def test_full_attribution_no_benchmark(self, pa, sample_returns):
        result = pa.full_attribution(sample_returns)
        assert "total_return" in result
        assert "alpha" not in result

    def test_empty_returns(self, pa):
        assert pa.full_attribution(pd.Series(dtype=float)) == {}

    def test_short_returns_alpha_beta(self, pa):
        result = pa.compute_alpha_beta(
            pd.Series([0.01, 0.02], index=[0, 1]),
            pd.Series([0.01, 0.02], index=[0, 1]),
        )
        assert result["alpha"] == 0.0


# ============================================================================
# DRAWDOWN CONTROLLER TESTS
# ============================================================================


class TestDrawdownController:
    def test_no_drawdown(self):
        ddc = DrawdownController()
        ddc.update_equity(100_000)
        assert ddc.current_drawdown == 0.0
        assert not ddc.should_reduce()

    def test_drawdown_below_threshold(self):
        ddc = DrawdownController(threshold=0.10)
        ddc.update_equity(100_000)
        ddc.update_equity(95_000)
        assert ddc.current_drawdown == 0.05
        assert not ddc.should_reduce()

    def test_drawdown_above_threshold(self):
        ddc = DrawdownController(threshold=0.08)
        ddc.update_equity(100_000)
        ddc.update_equity(91_000)
        assert ddc.should_reduce()

    def test_position_reduction(self):
        ddc = DrawdownController(threshold=0.08, reduce_factor=0.30)
        ddc.update_equity(100_000)
        ddc.update_equity(91_000)
        assert ddc.adjusted_position_size(100) == 70

    def test_no_reduction_when_ok(self):
        ddc = DrawdownController(threshold=0.08)
        ddc.update_equity(100_000)
        ddc.update_equity(95_000)
        assert ddc.adjusted_position_size(100) == 100

    def test_severe_drawdown_double_reduction(self):
        ddc = DrawdownController(threshold=0.08, reduce_factor=0.30)
        ddc.update_equity(100_000)
        ddc.update_equity(85_000)  # 15% DD > 1.5 * 8%
        adjusted = ddc.adjusted_position_size(100)
        assert adjusted < 70  # should reduce more than 30%

    def test_peak_equity_tracking(self):
        ddc = DrawdownController()
        ddc.update_equity(100_000)
        ddc.update_equity(110_000)
        ddc.update_equity(105_000)
        assert ddc.peak_equity == 110_000
        assert ddc.current_equity == 105_000

    def test_minimum_position(self):
        ddc = DrawdownController(threshold=0.01, reduce_factor=0.99)
        ddc.update_equity(100_000)
        ddc.update_equity(98_000)
        assert ddc.adjusted_position_size(1) >= 1


# ============================================================================
# MULTI-FACTOR RISK MODEL TESTS
# ============================================================================


class TestMultiFactorRiskModel:
    def test_compute_market_beta(self):
        mfr = MultiFactorRiskModel()
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="B")
        market = pd.Series(rng.normal(0.001, 0.01, 200), index=idx)
        stock = pd.Series(1.5 * market.values + rng.normal(0, 0.005, 200), index=idx)
        exposures = mfr.compute_factor_exposures(stock, market)
        assert "market_beta" in exposures
        assert abs(exposures["market_beta"] - 1.5) < 0.5

    def test_short_series(self):
        mfr = MultiFactorRiskModel()
        result = mfr.compute_factor_exposures(
            pd.Series([0.01, 0.02], index=[0, 1]),
            pd.Series([0.01, 0.02], index=[0, 1]),
        )
        assert result == {"market_beta": 0.0}

    def test_risk_contribution(self):
        mfr = MultiFactorRiskModel()
        weights = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.01], [0.01, 0.03]])
        rc = mfr.risk_contribution(weights, cov)
        assert len(rc) == 2
        assert all(np.isfinite(rc))

    def test_hrp_weights(self):
        mfr = MultiFactorRiskModel()
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = mfr.hierarchical_risk_parity(cov, 2)
        assert len(weights) == 2
        assert abs(weights.sum() - 1.0) < 1e-10
        assert weights[0] > weights[1]  # lower variance gets higher weight

    def test_zero_variance_hrp(self):
        mfr = MultiFactorRiskModel()
        cov = np.diag([0.0, 0.04])
        weights = mfr.hierarchical_risk_parity(cov, 2)
        assert all(np.isfinite(weights))


# ============================================================================
# SECTOR ROTATION DETECTOR TESTS
# ============================================================================


class TestSectorRotationDetector:
    def test_bullish_rotation(self):
        srd = SectorRotationDetector(threshold=0.70)
        signals = {"IT": ["BUY", "BUY", "BUY", "HOLD"]}
        result = srd.detect_rotation(signals)
        assert "IT" in result
        assert result["IT"]["direction"] == "BULLISH"

    def test_bearish_rotation(self):
        srd = SectorRotationDetector(threshold=0.70)
        signals = {"Banks": ["SELL", "SELL", "SELL", "HOLD"]}
        result = srd.detect_rotation(signals)
        assert "Banks" in result
        assert result["Banks"]["direction"] == "BEARISH"

    def test_no_rotation(self):
        srd = SectorRotationDetector(threshold=0.70)
        signals = {"Mixed": ["BUY", "SELL", "HOLD", "BUY"]}
        result = srd.detect_rotation(signals)
        assert "Mixed" not in result

    def test_empty_signals(self):
        srd = SectorRotationDetector()
        assert srd.detect_rotation({}) == {}

    def test_multiple_sectors(self):
        srd = SectorRotationDetector(threshold=0.60)
        signals = {
            "IT": ["BUY", "BUY", "BUY"],
            "Pharma": ["SELL", "SELL", "SELL"],
            "Auto": ["BUY", "HOLD", "SELL"],
        }
        result = srd.detect_rotation(signals)
        assert "IT" in result
        assert "Pharma" in result
        assert "Auto" not in result


# ============================================================================
# EXECUTION OPTIMIZER TESTS
# ============================================================================


class TestExecutionOptimizer:
    @pytest.fixture
    def exec_opt(self):
        return ExecutionOptimizer()

    def test_almgren_chriss_impact(self, exec_opt):
        result = exec_opt.almgren_chriss_impact(1000, 500_000, 0.02)
        assert result["total_cost_bps"] > 0
        assert result["temporary_impact"] > 0
        assert result["permanent_impact"] > 0

    def test_zero_volume_impact(self, exec_opt):
        result = exec_opt.almgren_chriss_impact(1000, 0, 0.02)
        assert result["total_cost_bps"] == 0.0

    def test_execution_schedule(self, exec_opt):
        schedule = exec_opt.optimal_execution_schedule(50_000, 200_000)
        assert len(schedule) > 0
        total = sum(s["quantity"] for s in schedule)
        assert total == 50_000

    def test_small_order_single_day(self, exec_opt):
        schedule = exec_opt.optimal_execution_schedule(1000, 1_000_000)
        assert len(schedule) == 1
        assert schedule[0]["quantity"] == 1000

    def test_large_order_multi_day(self, exec_opt):
        schedule = exec_opt.optimal_execution_schedule(100_000, 200_000, n_days=5)
        assert len(schedule) >= 2

    def test_empty_schedule(self, exec_opt):
        assert exec_opt.optimal_execution_schedule(0, 100_000) == []

    def test_time_stop_no_trigger(self, exec_opt):
        entry = datetime(2024, 1, 1)
        current = datetime(2024, 1, 10)
        assert not exec_opt.time_stop_check(entry, current, 100.0, 105.0)

    def test_time_stop_trigger(self, exec_opt):
        entry = datetime(2024, 1, 1)
        current = datetime(2024, 1, 20)
        assert exec_opt.time_stop_check(entry, current, 100.0, 100.5)

    def test_time_stop_moved_enough(self, exec_opt):
        entry = datetime(2024, 1, 1)
        current = datetime(2024, 1, 20)
        assert not exec_opt.time_stop_check(entry, current, 100.0, 110.0)


# ============================================================================
# EXPANDED FEATURE ENGINE TESTS
# ============================================================================


class TestExpandedFeatureEngine:
    @pytest.fixture
    def feature_engine(self):
        return FeatureEngine()

    @pytest.fixture
    def price_data(self):
        rng = np.random.default_rng(42)
        n = 300
        c = 100 + rng.normal(0, 1.2, n).cumsum()
        df = pd.DataFrame({
            "Open": c + rng.normal(0, 0.5, n),
            "High": c + abs(rng.normal(0, 1, n)) + 1,
            "Low": c - abs(rng.normal(0, 1, n)) - 1,
            "Close": c,
            "Volume": rng.integers(10000, 100000, n).astype(float),
        }, index=pd.date_range("2023-01-01", periods=n, freq="B"))
        return df

    def test_hull_moving_average(self, feature_engine, price_data):
        result = feature_engine.hull_moving_average(price_data["Close"], 9)
        assert len(result) == len(price_data)
        assert result.dropna().shape[0] > 0

    def test_aroon(self, feature_engine, price_data):
        up, down = feature_engine.aroon(price_data["High"], price_data["Low"])
        assert len(up) == len(price_data)
        assert len(down) == len(price_data)

    def test_elder_ray(self, feature_engine, price_data):
        bull, bear = feature_engine.elder_ray(
            price_data["High"], price_data["Low"], price_data["Close"]
        )
        assert len(bull) == len(price_data)

    def test_supertrend(self, feature_engine, price_data):
        result = feature_engine.supertrend(
            price_data["High"], price_data["Low"], price_data["Close"]
        )
        assert len(result) == len(price_data)

    def test_choppiness_index(self, feature_engine, price_data):
        result = feature_engine.choppiness_index(
            price_data["High"], price_data["Low"], price_data["Close"]
        )
        assert len(result) == len(price_data)

    def test_vortex_indicator(self, feature_engine, price_data):
        vi_plus, vi_minus = feature_engine.vortex_indicator(
            price_data["High"], price_data["Low"], price_data["Close"]
        )
        assert len(vi_plus) == len(price_data)

    def test_mass_index(self, feature_engine, price_data):
        result = feature_engine.mass_index(price_data["High"], price_data["Low"])
        assert len(result) == len(price_data)

    def test_corwin_schultz_spread(self, feature_engine, price_data):
        result = feature_engine.corwin_schultz_spread(
            price_data["High"], price_data["Low"]
        )
        assert len(result) == len(price_data)
        clean = result.dropna()
        assert (clean >= 0).all()

    def test_lempel_ziv_complexity(self, feature_engine, price_data):
        result = feature_engine.lempel_ziv_complexity(price_data["Close"])
        assert len(result) == len(price_data)

    def test_order_flow_imbalance(self, feature_engine, price_data):
        result = feature_engine.order_flow_imbalance(
            price_data["Open"], price_data["High"],
            price_data["Low"], price_data["Close"],
        )
        assert len(result) == len(price_data)

    def test_darvas_box(self, feature_engine, price_data):
        result = feature_engine.darvas_box(price_data["High"], price_data["Low"])
        assert "box_top" in result
        assert "breakout_up" in result

    def test_linear_regression_features(self, feature_engine, price_data):
        result = feature_engine.linear_regression_features(price_data["Close"])
        assert "slope" in result
        assert "r_squared" in result
        assert "deviation" in result

    def test_efficiency_ratio(self, feature_engine, price_data):
        result = feature_engine.efficiency_ratio(price_data["Close"])
        assert len(result) == len(price_data)

    def test_disparity_index(self, feature_engine, price_data):
        result = feature_engine.disparity_index(price_data["Close"], 14)
        assert len(result) == len(price_data)

    def test_coppock_curve(self, feature_engine, price_data):
        result = feature_engine.coppock_curve(price_data["Close"])
        assert len(result) == len(price_data)

    def test_compute_all_features_expanded(self, feature_engine, price_data):
        result = feature_engine.compute_all_features(price_data)
        assert len(result.columns) >= 150
        for col in ["hma_9", "aroon_up", "supertrend", "choppiness_14",
                     "mass_index", "lr_slope_20", "pmo", "darvas_breakout_up",
                     "order_flow_imbalance", "natr_14"]:
            assert col in result.columns, f"Missing feature: {col}"

    def test_lz_complexity_static(self):
        assert FeatureEngine._lz_complexity("") == 0
        assert FeatureEngine._lz_complexity("1111") >= 1
        assert FeatureEngine._lz_complexity("0101010101") >= 2

    def test_chaikin_volatility(self, feature_engine, price_data):
        result = feature_engine.chaikin_volatility(
            price_data["High"], price_data["Low"]
        )
        assert len(result) == len(price_data)

    def test_price_momentum_oscillator(self, feature_engine, price_data):
        result = feature_engine.price_momentum_oscillator(price_data["Close"])
        assert len(result) == len(price_data)

    def test_trend_intensity_index(self, feature_engine, price_data):
        result = feature_engine.trend_intensity_index(price_data["Close"])
        assert len(result) == len(price_data)


# ============================================================================
# INTEGRATION TEST: FULL PIPELINE WITH NEW COMPONENTS
# ============================================================================


class TestIntegrationNewComponents:
    def test_orchestrator_has_new_components(self):
        arch = NSE500AlphaArchitect(capital=100_000, state_dir="./test_int_state")
        assert hasattr(arch, "state_manager")
        assert hasattr(arch, "stress_tester")
        assert hasattr(arch, "walk_forward")
        assert hasattr(arch, "drift_detector")
        assert hasattr(arch, "cointegration")
        assert hasattr(arch, "corp_actions")
        assert hasattr(arch, "perf_analytics")
        assert hasattr(arch, "dd_controller")
        assert hasattr(arch, "factor_model")
        assert hasattr(arch, "sector_rotation")
        assert hasattr(arch, "exec_optimizer")
        import shutil
        shutil.rmtree("./test_int_state", ignore_errors=True)

    def test_pipeline_includes_drift_stress(self):
        arch = NSE500AlphaArchitect(capital=100_000, state_dir="./test_pipe_state")
        summary = arch.run_full_pipeline()
        assert "drift_detection" in summary
        assert "stress_test" in summary
        assert "drawdown" in summary
        import shutil
        shutil.rmtree("./test_pipe_state", ignore_errors=True)
        shutil.rmtree("./state", ignore_errors=True)
