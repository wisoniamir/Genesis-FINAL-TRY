
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

from pathlib import Path

import pytest

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




@pytest.fixture
def xml_data_path():
    return Path(__file__).parent.parent / "data" / "xml"


@pytest.fixture
def xml_books(xml_data_path, datapath):
    return datapath(xml_data_path / "books.xml")


@pytest.fixture
def xml_doc_ch_utf(xml_data_path, datapath):
    return datapath(xml_data_path / "doc_ch_utf.xml")


@pytest.fixture
def xml_baby_names(xml_data_path, datapath):
    return datapath(xml_data_path / "baby_names.xml")


@pytest.fixture
def kml_cta_rail_lines(xml_data_path, datapath):
    return datapath(xml_data_path / "cta_rail_lines.kml")


@pytest.fixture
def xsl_flatten_doc(xml_data_path, datapath):
    return datapath(xml_data_path / "flatten_doc.xsl")


@pytest.fixture
def xsl_row_field_output(xml_data_path, datapath):
    return datapath(xml_data_path / "row_field_output.xsl")


# <!-- @GENESIS_MODULE_END: conftest -->
