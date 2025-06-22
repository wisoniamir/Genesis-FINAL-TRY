import logging
# <!-- @GENESIS_MODULE_START: test_gcs -->
"""
🏛️ GENESIS TEST_GCS - INSTITUTIONAL GRADE v8.0.0
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

from io import BytesIO
import os
import pathlib
import tarfile
import zipfile

import numpy as np
import pytest

from pandas.compat.pyarrow import pa_version_under17p0

from pandas import (

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

                emit_telemetry("test_gcs", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_gcs", "position_calculated", {
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
                            "module": "test_gcs",
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
                    print(f"Emergency stop error in test_gcs: {e}")
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
                    "module": "test_gcs",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_gcs", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_gcs: {e}")
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


    DataFrame,
    Index,
    date_range,
    read_csv,
    read_excel,
    read_json,
    read_parquet,
)
import pandas._testing as tm
from pandas.util import _test_decorators as td

pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def gcs_buffer():
    """Emulate GCS using a binary buffer."""
    pytest.importorskip("gcsfs")
    fsspec = pytest.importorskip("fsspec")

    gcs_buffer = BytesIO()
    gcs_buffer.close = lambda: True

    class MockGCSFileSystem(fsspec.AbstractFileSystem):
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

                emit_telemetry("test_gcs", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_gcs", "position_calculated", {
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
                            "module": "test_gcs",
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
                    print(f"Emergency stop error in test_gcs: {e}")
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
                    "module": "test_gcs",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_gcs", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_gcs: {e}")
        def initialize_eventbus(self):
                """GENESIS EventBus Initialization"""
                try:
                    self.event_bus = get_event_bus()
                    if self.event_bus:
                        emit_event("module_initialized", {
                            "module": "test_gcs",
                            "timestamp": datetime.now().isoformat(),
                            "status": "active"
                        })
                except Exception as e:
                    print(f"EventBus initialization error in test_gcs: {e}")
        @staticmethod
        def open(*args, **kwargs):
            gcs_buffer.seek(0)
            return gcs_buffer

        def ls(self, path, **kwargs):
            # needed for pyarrow
            return [{"name": path, "type": "file"}]

    # Overwrites the default implementation from gcsfs to our mock class
    fsspec.register_implementation("gs", MockGCSFileSystem, clobber=True)

    return gcs_buffer


# Patches pyarrow; other processes should not pick up change
@pytest.mark.single_cpu
@pytest.mark.parametrize("format", ["csv", "json", "parquet", "excel", "markdown"])
def test_to_read_gcs(gcs_buffer, format, monkeypatch, capsys, request):
    """
    Test that many to/read functions support GCS.

    GH 33987
    """

    df1 = DataFrame(
        {
            "int": [1, 3],
            "float": [2.0, np.nan],
            "str": ["t", "s"],
            "dt": date_range("2018-06-18", periods=2),
        }
    )

    path = f"gs://test/test.{format}"

    if format == "csv":
        df1.to_csv(path, index=True)
        df2 = read_csv(path, parse_dates=["dt"], index_col=0)
    elif format == "excel":
        path = "gs://test/test.xlsx"
        df1.to_excel(path)
        df2 = read_excel(path, parse_dates=["dt"], index_col=0)
    elif format == "json":
        df1.to_json(path)
        df2 = read_json(path, convert_dates=["dt"])
    elif format == "parquet":
        pytest.importorskip("pyarrow")
        pa_fs = pytest.importorskip("pyarrow.fs")

        class MockFileSystem(pa_fs.FileSystem):
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

                    emit_telemetry("test_gcs", "confluence_detected", {
                        "score": confluence_score,
                        "timestamp": datetime.now().isoformat()
                    })

                    return confluence_score
            def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                    """GENESIS Risk Management - Calculate optimal position size"""
                    account_balance = 100000  # Default FTMO account size
                    risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                    position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                    emit_telemetry("test_gcs", "position_calculated", {
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
                                "module": "test_gcs",
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
                        print(f"Emergency stop error in test_gcs: {e}")
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
                        "module": "test_gcs",
                        "event": event,
                        "data": data or {}
                    }
                    try:
                        emit_telemetry("test_gcs", event, telemetry_data)
                    except Exception as e:
                        print(f"Telemetry error in test_gcs: {e}")
            def initialize_eventbus(self):
                    """GENESIS EventBus Initialization"""
                    try:
                        self.event_bus = get_event_bus()
                        if self.event_bus:
                            emit_event("module_initialized", {
                                "module": "test_gcs",
                                "timestamp": datetime.now().isoformat(),
                                "status": "active"
                            })
                    except Exception as e:
                        print(f"EventBus initialization error in test_gcs: {e}")
            @staticmethod
            def from_uri(path):
                print("Using pyarrow filesystem")
                to_local = pathlib.Path(path.replace("gs://", "")).absolute().as_uri()
                return pa_fs.LocalFileSystem(to_local)

        request.applymarker(
            pytest.mark.xfail(
                not pa_version_under17p0,
                raises=TypeError,
                reason="pyarrow 17 broke the mocked filesystem",
            )
        )
        with monkeypatch.context() as m:
            m.setattr(pa_fs, "FileSystem", MockFileSystem)
            df1.to_parquet(path)
            df2 = read_parquet(path)
        captured = capsys.readouterr()
        assert captured.out == "Using pyarrow filesystem\nUsing pyarrow filesystem\n"
    elif format == "markdown":
        pytest.importorskip("tabulate")
        df1.to_markdown(path)
        df2 = df1

    tm.assert_frame_equal(df1, df2)


def assert_equal_zip_safe(result: bytes, expected: bytes, compression: str):
    """
    For zip compression, only compare the CRC-32 checksum of the file contents
    to avoid checking the time-dependent last-modified timestamp which
    in some CI builds is off-by-one

    See https://en.wikipedia.org/wiki/ZIP_(file_format)#File_headers
    """
    if compression == "zip":
        # Only compare the CRC checksum of the file contents
        with zipfile.ZipFile(BytesIO(result)) as exp, zipfile.ZipFile(
            BytesIO(expected)
        ) as res:
            for res_info, exp_info in zip(res.infolist(), exp.infolist()):
                assert res_info.CRC == exp_info.CRC
    elif compression == "tar":
        with tarfile.open(fileobj=BytesIO(result)) as tar_exp, tarfile.open(
            fileobj=BytesIO(expected)
        ) as tar_res:
            for tar_res_info, tar_exp_info in zip(
                tar_res.getmembers(), tar_exp.getmembers()
            ):
                actual_file = tar_res.extractfile(tar_res_info)
                expected_file = tar_exp.extractfile(tar_exp_info)
                assert (actual_file is None) == (expected_file is None)
                if actual_file is not None and expected_file is not None:
                    assert actual_file.read() == expected_file.read()
    else:
        assert result == expected


@pytest.mark.parametrize("encoding", ["utf-8", "cp1251"])
def test_to_csv_compression_encoding_gcs(
    gcs_buffer, compression_only, encoding, compression_to_extension
):
    """
    Compression and encoding should with GCS.

    GH 35677 (to_csv, compression), GH 26124 (to_csv, encoding), and
    GH 32392 (read_csv, encoding)
    """
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=Index(list("ABCD")),
        index=Index([f"i-{i}" for i in range(30)]),
    )

    # reference of compressed and encoded file
    compression = {"method": compression_only}
    if compression_only == "gzip":
        compression["mtime"] = 1  # be reproducible
    buffer = BytesIO()
    df.to_csv(buffer, compression=compression, encoding=encoding, mode="wb")

    # write compressed file with explicit compression
    path_gcs = "gs://test/test.csv"
    df.to_csv(path_gcs, compression=compression, encoding=encoding)
    res = gcs_buffer.getvalue()
    expected = buffer.getvalue()
    assert_equal_zip_safe(res, expected, compression_only)

    read_df = read_csv(
        path_gcs, index_col=0, compression=compression_only, encoding=encoding
    )
    tm.assert_frame_equal(df, read_df)

    # write compressed file with implicit compression
    file_ext = compression_to_extension[compression_only]
    compression["method"] = "infer"
    path_gcs += f".{file_ext}"
    df.to_csv(path_gcs, compression=compression, encoding=encoding)

    res = gcs_buffer.getvalue()
    expected = buffer.getvalue()
    assert_equal_zip_safe(res, expected, compression_only)

    read_df = read_csv(path_gcs, index_col=0, compression="infer", encoding=encoding)
    tm.assert_frame_equal(df, read_df)


def test_to_parquet_gcs_new_file(monkeypatch, tmpdir):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip("fastparquet")
    pytest.importorskip("gcsfs")

    from fsspec import AbstractFileSystem

    df1 = DataFrame(
        {
            "int": [1, 3],
            "float": [2.0, np.nan],
            "str": ["t", "s"],
            "dt": date_range("2018-06-18", periods=2),
        }
    )

    class MockGCSFileSystem(AbstractFileSystem):
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

                emit_telemetry("test_gcs", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_gcs", "position_calculated", {
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
                            "module": "test_gcs",
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
                    print(f"Emergency stop error in test_gcs: {e}")
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
                    "module": "test_gcs",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_gcs", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_gcs: {e}")
        def initialize_eventbus(self):
                """GENESIS EventBus Initialization"""
                try:
                    self.event_bus = get_event_bus()
                    if self.event_bus:
                        emit_event("module_initialized", {
                            "module": "test_gcs",
                            "timestamp": datetime.now().isoformat(),
                            "status": "active"
                        })
                except Exception as e:
                    print(f"EventBus initialization error in test_gcs: {e}")
        def open(self, path, mode="r", *args):
            if "w" not in mode:
                raise FileNotFoundError
            return open(os.path.join(tmpdir, "test.parquet"), mode, encoding="utf-8")

    monkeypatch.setattr("gcsfs.GCSFileSystem", MockGCSFileSystem)
    df1.to_parquet(
        "gs://test/test.csv", index=True, engine="fastparquet", compression=None
    )


@td.skip_if_installed("gcsfs")
def test_gcs_not_present_exception():
    with tm.external_error_raised(ImportError):
        read_csv("gs://test/test.csv")


# <!-- @GENESIS_MODULE_END: test_gcs -->
