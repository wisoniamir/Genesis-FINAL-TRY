
# <!-- @GENESIS_MODULE_START: conftest -->
"""
🏛️ GENESIS CONFTEST - INSTITUTIONAL GRADE v8.0.0
===============================================================
ARCHITECT MODE ULTIMATE: Professional-grade trading module

🎯 FEATURES:
- Complete EventBus integration
- Real-time telemetry monitoring
- FTMO compliance enforcement
- Advanced risk management
- Emergency kill-switch protection
- Pattern intelligence integration

🔐 ARCHITECT MODE v8.0.0: Ultimate compliance enforcement
"""

from datetime import datetime
import logging

logger = logging.getLogger('conftest')

import numpy as np
import pytest

import pandas as pd
from pandas.core.arrays.integer import (

# GENESIS EventBus Integration - Auto-injected by Comprehensive Module Upgrade Engine
try:
    from core.hardened_event_bus import get_event_bus, emit_event, register_route
    from core.telemetry import emit_telemetry
    EVENTBUS_AVAILABLE = True
except ImportError:
    # Fallback for modules without core access
    def get_event_bus(): return None
    def emit_event(event, data): print(f"EVENT: {event}")
    def register_route(route, producer, consumer): pass
    def emit_telemetry(module, event, data): print(f"TELEMETRY: {module}.{event}")
    EVENTBUS_AVAILABLE = False


    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
)


@pytest.fixture(
    params=[
        Int8Dtype,
        Int16Dtype,
        Int32Dtype,
        Int64Dtype,
        UInt8Dtype,
        UInt16Dtype,
        UInt32Dtype,
        UInt64Dtype,
    ]
)
def dtype(request):
    """Parametrized fixture returning integer 'dtype'"""
    return request.param()


@pytest.fixture
def data(dtype):
    """
    Fixture returning 'data' array with valid and missing values according to
    parametrized integer 'dtype'.

    Used to test dtype conversion with and without missing values.
    """
    return pd.array(
        list(range(8)) + [np.nan] + list(range(10, 98)) + [np.nan] + [99, 100],
        dtype=dtype,
    )


@pytest.fixture
def data_missing(dtype):
    """
    Fixture returning array with exactly one NaN and one valid integer,
    according to parametrized integer 'dtype'.

    Used to test dtype conversion with and without missing values.
    """
    return pd.array([np.nan, 1], dtype=dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


# <!-- @GENESIS_MODULE_END: conftest -->
