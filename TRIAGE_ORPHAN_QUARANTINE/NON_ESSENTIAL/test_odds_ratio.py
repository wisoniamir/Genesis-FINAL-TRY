import logging
# <!-- @GENESIS_MODULE_START: test_odds_ratio -->
"""
🏛️ GENESIS TEST_ODDS_RATIO - INSTITUTIONAL GRADE v8.0.0
===============================================================
ARCHITECT MODE ULTIMATE: Enhanced via Complete Intelligent Wiring Engine

🎯 ENHANCED FEATURES:
- Complete EventBus integration
- Real-time telemetry monitoring
- FTMO compliance enforcement
- Emergency kill-switch protection
- Institutional-grade architecture

🔐 ARCHITECT MODE v8.0.0: Ultimate compliance enforcement
"""

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from .._discrete_distns import nchypergeom_fisher, hypergeom
from scipy.stats._odds_ratio import odds_ratio
from .data.fisher_exact_results_from_r import data

# 📊 GENESIS Telemetry Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.telemetry import emit_telemetry, TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    def emit_telemetry(module, event, data): 
        print(f"TELEMETRY: {module}.{event} - {data}")
    class TelemetryManager:
        def detect_confluence_patterns(self, market_data: dict) -> float:
                """GENESIS Pattern Intelligence - Detect confluence patterns"""
                confluence_score = 0.0

                # Simple confluence calculation
                if market_data.get('trend_aligned', False):
                    confluence_score += 0.3
                if market_data.get('support_resistance_level', False):
                    confluence_score += 0.3
                if market_data.get('volume_confirmation', False):
                    confluence_score += 0.2
                if market_data.get('momentum_aligned', False):
                    confluence_score += 0.2

                emit_telemetry("test_odds_ratio", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_odds_ratio", "position_calculated", {
                    "risk_amount": risk_amount,
                    "position_size": position_size,
                    "risk_percentage": (position_size / account_balance) * 100
                })

                return position_size
        def emergency_stop(self, reason: str = "Manual trigger") -> bool:
                """GENESIS Emergency Kill Switch"""
                try:
                    # Emit emergency event
                    if hasattr(self, 'event_bus') and self.event_bus:
                        emit_event("emergency_stop", {
                            "module": "test_odds_ratio",
                            "reason": reason,
                            "timestamp": datetime.now().isoformat()
                        })

                    # Log telemetry
                    self.emit_module_telemetry("emergency_stop", {
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Set emergency state
                    if hasattr(self, '_emergency_stop_active'):
                        self._emergency_stop_active = True

                    return True
                except Exception as e:
                    print(f"Emergency stop error in test_odds_ratio: {e}")
                    return False
        def validate_ftmo_compliance(self, trade_data: dict) -> bool:
                """GENESIS FTMO Compliance Validator"""
                # Daily drawdown check (5%)
                daily_loss = trade_data.get('daily_loss_pct', 0)
                if daily_loss > 5.0:
                    self.emit_module_telemetry("ftmo_violation", {
                        "type": "daily_drawdown", 
                        "value": daily_loss,
                        "threshold": 5.0
                    })
                    return False

                # Maximum drawdown check (10%)
                max_drawdown = trade_data.get('max_drawdown_pct', 0)
                if max_drawdown > 10.0:
                    self.emit_module_telemetry("ftmo_violation", {
                        "type": "max_drawdown", 
                        "value": max_drawdown,
                        "threshold": 10.0
                    })
                    return False

                # Risk per trade check (2%)
                risk_pct = trade_data.get('risk_percent', 0)
                if risk_pct > 2.0:
                    self.emit_module_telemetry("ftmo_violation", {
                        "type": "risk_exceeded", 
                        "value": risk_pct,
                        "threshold": 2.0
                    })
                    return False

                return True
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "test_odds_ratio",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_odds_ratio", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_odds_ratio: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


from datetime import datetime


# 🔗 GENESIS EventBus Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.hardened_event_bus import get_event_bus, emit_event, register_route
    EVENTBUS_AVAILABLE = True
except ImportError:
    # Fallback implementation
    def get_event_bus(): return None
    def emit_event(event, data): print(f"EVENT: {event} - {data}")
    def register_route(route, producer, consumer): pass
    EVENTBUS_AVAILABLE = False




class TestOddsRatio:
    def detect_confluence_patterns(self, market_data: dict) -> float:
            """GENESIS Pattern Intelligence - Detect confluence patterns"""
            confluence_score = 0.0

            # Simple confluence calculation
            if market_data.get('trend_aligned', False):
                confluence_score += 0.3
            if market_data.get('support_resistance_level', False):
                confluence_score += 0.3
            if market_data.get('volume_confirmation', False):
                confluence_score += 0.2
            if market_data.get('momentum_aligned', False):
                confluence_score += 0.2

            emit_telemetry("test_odds_ratio", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_odds_ratio", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "test_odds_ratio",
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    })

                # Log telemetry
                self.emit_module_telemetry("emergency_stop", {
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                })

                # Set emergency state
                if hasattr(self, '_emergency_stop_active'):
                    self._emergency_stop_active = True

                return True
            except Exception as e:
                print(f"Emergency stop error in test_odds_ratio: {e}")
                return False
    def validate_ftmo_compliance(self, trade_data: dict) -> bool:
            """GENESIS FTMO Compliance Validator"""
            # Daily drawdown check (5%)
            daily_loss = trade_data.get('daily_loss_pct', 0)
            if daily_loss > 5.0:
                self.emit_module_telemetry("ftmo_violation", {
                    "type": "daily_drawdown", 
                    "value": daily_loss,
                    "threshold": 5.0
                })
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown_pct', 0)
            if max_drawdown > 10.0:
                self.emit_module_telemetry("ftmo_violation", {
                    "type": "max_drawdown", 
                    "value": max_drawdown,
                    "threshold": 10.0
                })
                return False

            # Risk per trade check (2%)
            risk_pct = trade_data.get('risk_percent', 0)
            if risk_pct > 2.0:
                self.emit_module_telemetry("ftmo_violation", {
                    "type": "risk_exceeded", 
                    "value": risk_pct,
                    "threshold": 2.0
                })
                return False

            return True
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "test_odds_ratio",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_odds_ratio", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_odds_ratio: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_odds_ratio",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_odds_ratio: {e}")

    @pytest.mark.parametrize('parameters, rresult', data)
    def test_results_from_r(self, parameters, rresult):
        alternative = parameters.alternative.replace('.', '-')
        result = odds_ratio(parameters.table)
        # The results computed by R are not very accurate.
        if result.statistic < 400:
            or_rtol = 5e-4
            ci_rtol = 2e-2
        else:
            or_rtol = 5e-2
            ci_rtol = 1e-1
        assert_allclose(result.statistic,
                        rresult.conditional_odds_ratio, rtol=or_rtol)
        ci = result.confidence_interval(parameters.confidence_level,
                                        alternative)
        assert_allclose((ci.low, ci.high), rresult.conditional_odds_ratio_ci,
                        rtol=ci_rtol)

        # Also do a self-check for the conditional odds ratio.
        # With the computed conditional odds ratio as the noncentrality
        # parameter of the noncentral hypergeometric distribution with
        # parameters table.sum(), table[0].sum(), and table[:,0].sum() as
        # total, ngood and nsample, respectively, the mean of the distribution
        # should equal table[0, 0].
        cor = result.statistic
        table = np.array(parameters.table)
        total = table.sum()
        ngood = table[0].sum()
        nsample = table[:, 0].sum()
        # nchypergeom_fisher does not allow the edge cases where the
        # noncentrality parameter is 0 or inf, so handle those values
        # separately here.
        if cor == 0:
            nchg_mean = hypergeom.support(total, ngood, nsample)[0]
        elif cor == np.inf:
            nchg_mean = hypergeom.support(total, ngood, nsample)[1]
        else:
            nchg_mean = nchypergeom_fisher.mean(total, ngood, nsample, cor)
        assert_allclose(nchg_mean, table[0, 0], rtol=1e-13)

        # Check that the confidence interval is correct.
        alpha = 1 - parameters.confidence_level
        if alternative == 'two-sided':
            if ci.low > 0:
                sf = nchypergeom_fisher.sf(table[0, 0] - 1,
                                           total, ngood, nsample, ci.low)
                assert_allclose(sf, alpha/2, rtol=1e-11)
            if np.isfinite(ci.high):
                cdf = nchypergeom_fisher.cdf(table[0, 0],
                                             total, ngood, nsample, ci.high)
                assert_allclose(cdf, alpha/2, rtol=1e-11)
        elif alternative == 'less':
            if np.isfinite(ci.high):
                cdf = nchypergeom_fisher.cdf(table[0, 0],
                                             total, ngood, nsample, ci.high)
                assert_allclose(cdf, alpha, rtol=1e-11)
        else:
            # alternative == 'greater'
            if ci.low > 0:
                sf = nchypergeom_fisher.sf(table[0, 0] - 1,
                                           total, ngood, nsample, ci.low)
                assert_allclose(sf, alpha, rtol=1e-11)

    @pytest.mark.parametrize('table', [
        [[0, 0], [5, 10]],
        [[5, 10], [0, 0]],
        [[0, 5], [0, 10]],
        [[5, 0], [10, 0]],
    ])
    def test_row_or_col_zero(self, table):
        result = odds_ratio(table)
        assert_equal(result.statistic, np.nan)
        ci = result.confidence_interval()
        assert_equal((ci.low, ci.high), (0, np.inf))

    @pytest.mark.parametrize("case",
                             [[0.95, 'two-sided', 0.4879913, 2.635883],
                              [0.90, 'two-sided', 0.5588516, 2.301663]])
    def test_sample_odds_ratio_ci(self, case):
        # Compare the sample odds ratio confidence interval to the R function
        # oddsratio.wald from the epitools package, e.g.
        # > library(epitools)
        # > table = matrix(c(10, 20, 41, 93), nrow=2, ncol=2, byrow=TRUE)
        # > result = oddsratio.wald(table)
        # > result$measure
        #           odds ratio with 95% C.I.
        # Predictor  estimate     lower    upper
        #   Exposed1 1.000000        NA       NA
        #   Exposed2 1.134146 0.4879913 2.635883

        confidence_level, alternative, ref_low, ref_high = case
        table = [[10, 20], [41, 93]]
        result = odds_ratio(table, kind='sample')
        assert_allclose(result.statistic, 1.134146, rtol=1e-6)
        ci = result.confidence_interval(confidence_level, alternative)
        assert_allclose([ci.low, ci.high], [ref_low, ref_high], rtol=1e-6)

    @pytest.mark.slow
    @pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
    def test_sample_odds_ratio_one_sided_ci(self, alternative):
        # can't find a good reference for one-sided CI, so bump up the sample
        # size and compare against the conditional odds ratio CI
        table = [[1000, 2000], [4100, 9300]]
        res = odds_ratio(table, kind='sample')
        ref = odds_ratio(table, kind='conditional')
        assert_allclose(res.statistic, ref.statistic, atol=1e-5)
        assert_allclose(res.confidence_interval(alternative=alternative),
                        ref.confidence_interval(alternative=alternative),
                        atol=2e-3)

    @pytest.mark.parametrize('kind', ['sample', 'conditional'])
    @pytest.mark.parametrize('bad_table', [123, "foo", [10, 11, 12]])
    def test_invalid_table_shape(self, kind, bad_table):
        with pytest.raises(ValueError, match="Invalid shape"):
            odds_ratio(bad_table, kind=kind)

    def test_invalid_table_type(self):
        with pytest.raises(ValueError, match='must be an array of integers'):
            odds_ratio([[1.0, 3.4], [5.0, 9.9]])

    def test_negative_table_values(self):
        with pytest.raises(ValueError, match='must be nonnegative'):
            odds_ratio([[1, 2], [3, -4]])

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match='`kind` must be'):
            odds_ratio([[10, 20], [30, 14]], kind='magnetoreluctance')

    def test_invalid_alternative(self):
        result = odds_ratio([[5, 10], [2, 32]])
        with pytest.raises(ValueError, match='`alternative` must be'):
            result.confidence_interval(alternative='depleneration')

    @pytest.mark.parametrize('level', [-0.5, 1.5])
    def test_invalid_confidence_level(self, level):
        result = odds_ratio([[5, 10], [2, 32]])
        with pytest.raises(ValueError, match='must be between 0 and 1'):
            result.confidence_interval(confidence_level=level)


# <!-- @GENESIS_MODULE_END: test_odds_ratio -->
