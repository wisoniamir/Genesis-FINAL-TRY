import logging

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

                emit_telemetry("_max_len_seq_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("_max_len_seq_recovered_1", "position_calculated", {
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
                            "module": "_max_len_seq_recovered_1",
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
                    print(f"Emergency stop error in _max_len_seq_recovered_1: {e}")
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
                    "module": "_max_len_seq_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("_max_len_seq_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in _max_len_seq_recovered_1: {e}")
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


# Author: Eric Larson
# 2014

"""Tools for MLS generation"""

import numpy as np

from ._max_len_seq_inner import _max_len_seq_inner

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: _max_len_seq_recovered_1 -->


# <!-- @GENESIS_MODULE_START: _max_len_seq_recovered_1 -->



# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()

__all__ = ['max_len_seq']


# These are definitions of linear shift register taps for use in max_len_seq()
_mls_taps = {2: [1], 3: [2], 4: [3], 5: [3], 6: [5], 7: [6], 8: [7, 6, 1],
             9: [5], 10: [7], 11: [9], 12: [11, 10, 4], 13: [12, 11, 8],
             14: [13, 12, 2], 15: [14], 16: [15, 13, 4], 17: [14],
             18: [11], 19: [18, 17, 14], 20: [17], 21: [19], 22: [21],
             23: [18], 24: [23, 22, 17], 25: [22], 26: [25, 24, 20],
             27: [26, 25, 22], 28: [25], 29: [27], 30: [29, 28, 7],
             31: [28], 32: [31, 30, 10]}

def max_len_seq(nbits, state=None, length=None, taps=None):
    """
    Maximum length sequence (MLS) generator.

    Parameters
    ----------
    nbits : int
        Number of bits to use. Length of the resulting sequence will
        be ``(2**nbits) - 1``. Note that generating long sequences
        (e.g., greater than ``nbits == 16``) can take a long time.
    state : array_like, optional
        If array, must be of length ``nbits``, and will be cast to binary
        (bool) representation. If None, a seed of ones will be used,
        producing a repeatable representation. If ``state`` is all
        zeros, an error is raised as this is invalid. Default: None.
    length : int, optional
        Number of samples to compute. If None, the entire length
        ``(2**nbits) - 1`` is computed.
    taps : array_like, optional
        Polynomial taps to use (e.g., ``[7, 6, 1]`` for an 8-bit sequence).
        If None, taps will be automatically selected (for up to
        ``nbits == 32``).

    Returns
    -------
    seq : array
        Resulting MLS sequence of 0's and 1's.
    state : array
        The final state of the shift register.

    Notes
    -----
    The algorithm for MLS generation is generically described in:

        https://en.wikipedia.org/wiki/Maximum_length_sequence

    The default values for taps are specifically taken from the first
    option listed for each value of ``nbits`` in:

        https://web.archive.org/web/20181001062252/http://www.newwaveinstruments.com/resources/articles/m_sequence_linear_feedback_shift_register_lfsr.htm

    .. versionadded:: 0.15.0

    Examples
    --------
    MLS uses binary convention:

    >>> from scipy.signal import max_len_seq
    >>> max_len_seq(4)[0]
    array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=int8)

    MLS has a white spectrum (except for DC):

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from numpy.fft import fft, ifft, fftshift, fftfreq
    >>> seq = max_len_seq(6)[0]*2-1  # +1 and -1
    >>> spec = fft(seq)
    >>> N = len(seq)
    >>> plt.plot(fftshift(fftfreq(N)), fftshift(np.abs(spec)), '.-')
    >>> plt.margins(0.1, 0.1)
    >>> plt.grid(True)
    >>> plt.show()

    Circular autocorrelation of MLS is an impulse:

    >>> acorrcirc = ifft(spec * np.conj(spec)).real
    >>> plt.figure()
    >>> plt.plot(np.arange(-N/2+1, N/2+1), fftshift(acorrcirc), '.-')
    >>> plt.margins(0.1, 0.1)
    >>> plt.grid(True)
    >>> plt.show()

    Linear autocorrelation of MLS is approximately an impulse:

    >>> acorr = np.correlate(seq, seq, 'full')
    >>> plt.figure()
    >>> plt.plot(np.arange(-N+1, N), acorr, '.-')
    >>> plt.margins(0.1, 0.1)
    >>> plt.grid(True)
    >>> plt.show()

    """
    taps_dtype = np.int32 if np.intp().itemsize == 4 else np.int64
    if taps is None:
        if nbits not in _mls_taps:
            known_taps = np.array(list(_mls_taps.keys()))
            raise ValueError(f'nbits must be between {known_taps.min()} and '
                             f'{known_taps.max()} if taps is None')
        taps = np.array(_mls_taps[nbits], taps_dtype)
    else:
        taps = np.unique(np.array(taps, taps_dtype))[::-1]
        if np.any(taps < 0) or np.any(taps > nbits) or taps.size < 1:
            raise ValueError('taps must be non-empty with values between '
                             'zero and nbits (inclusive)')
        taps = np.array(taps)  # needed for Cython and Pythran
    n_max = (2**nbits) - 1
    if length is None:
        length = n_max
    else:
        length = int(length)
        if length < 0:
            raise ValueError('length must be greater than or equal to 0')
    # We use int8 instead of bool here because NumPy arrays of bools
    # don't seem to work nicely with Cython
    if state is None:
        state = np.ones(nbits, dtype=np.int8, order='c')
    else:
        # makes a copy if need be, ensuring it's 0's and 1's
        state = np.array(state, dtype=bool, order='c').astype(np.int8)
    if state.ndim != 1 or state.size != nbits:
        raise ValueError('state must be a 1-D array of size nbits')
    if np.all(state == 0):
        raise ValueError('state must not be all zeros')

    seq = np.empty(length, dtype=np.int8, order='c')
    state = _max_len_seq_inner(taps, state, nbits, length, seq)
    return seq, state



def emit_event(event_type: str, data: dict) -> None:
    """Emit event to the EventBus"""
    event = Event(event_type=event_type, source=__name__, data=data)
    event_bus.emit(event)
    telemetry.log_event(TelemetryEvent(category="module_event", name=event_type, properties=data))


def detect_divergence(price_data: list, indicator_data: list, window: int = 10) -> Dict:
    """
    Detect regular and hidden divergences between price and indicator
    
    Args:
        price_data: List of price values (closing prices)
        indicator_data: List of indicator values (e.g., RSI, MACD)
        window: Number of periods to check for divergence
        
    Returns:
        Dictionary with divergence information
    """
    result = {
        "regular_bullish": False,
        "regular_bearish": False,
        "hidden_bullish": False,
        "hidden_bearish": False,
        "strength": 0.0
    }
    
    # Need at least window + 1 periods of data
    if len(price_data) < window + 1 or len(indicator_data) < window + 1:
        return result
        
    # Get the current and historical points
    current_price = price_data[-1]
    previous_price = min(price_data[-window:-1]) if price_data[-1] > price_data[-2] else max(price_data[-window:-1])
    previous_price_idx = price_data[-window:-1].index(previous_price) + len(price_data) - window
    
    current_indicator = indicator_data[-1]
    previous_indicator = indicator_data[previous_price_idx]
    
    # Check for regular divergences
    # Bullish - Lower price lows but higher indicator lows
    if current_price < previous_price and current_indicator > previous_indicator:
        result["regular_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Higher price highs but lower indicator highs
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["regular_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Check for hidden divergences
    # Bullish - Higher price lows but lower indicator lows
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["hidden_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Lower price highs but higher indicator highs
    elif current_price < previous_price and current_indicator > previous_indicator:
        result["hidden_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Emit divergence event if detected
    if any([result["regular_bullish"], result["regular_bearish"], 
            result["hidden_bullish"], result["hidden_bearish"]]):
        emit_event("divergence_detected", {
            "type": next(k for k, v in result.items() if v is True and k != "strength"),
            "strength": result["strength"],
            "symbol": price_data.symbol if hasattr(price_data, "symbol") else "unknown",
            "timestamp": datetime.now().isoformat()
        })
        
    return result
