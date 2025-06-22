import logging
import sys
from pathlib import Path

# <!-- @GENESIS_MODULE_START: test_aix -->
"""
🏛️ GENESIS TEST_AIX - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("test_aix", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_aix", "position_calculated", {
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
                            "module": "test_aix",
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
                    print(f"Emergency stop error in test_aix: {e}")
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
                    "module": "test_aix",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_aix", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_aix: {e}")
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


#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'
# Copyright (c) 2017, Arnon Yaari
# All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""AIX specific tests."""

import re

import psutil
from psutil import AIX
from psutil.tests import PsutilTestCase
from psutil.tests import pytest
from psutil.tests import sh


@pytest.mark.skipif(not AIX, reason="AIX only")
class AIXSpecificTestCase(PsutilTestCase):
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

            emit_telemetry("test_aix", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_aix", "position_calculated", {
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
                        "module": "test_aix",
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
                print(f"Emergency stop error in test_aix: {e}")
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
                "module": "test_aix",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_aix", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_aix: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_aix",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_aix: {e}")
    def test_virtual_memory(self):
        out = sh('/usr/bin/svmon -O unit=KB')
        re_pattern = r"memory\s*"
        for field in [
            "size",
            "inuse",
            "free",
            "pin",
            "virtual",
            "available",
            "mmode",
        ]:
            re_pattern += rf"(?P<{field}>\S+)\s+"
        matchobj = re.search(re_pattern, out)

        assert matchobj is not None

        KB = 1024
        total = int(matchobj.group("size")) * KB
        available = int(matchobj.group("available")) * KB
        used = int(matchobj.group("inuse")) * KB
        free = int(matchobj.group("free")) * KB

        psutil_result = psutil.virtual_memory()

        # TOLERANCE_SYS_MEM from psutil.tests is not enough. For some reason
        # we're seeing differences of ~1.2 MB. 2 MB is still a good tolerance
        # when compared to GBs.
        TOLERANCE_SYS_MEM = 2 * KB * KB  # 2 MB
        assert psutil_result.total == total
        assert abs(psutil_result.used - used) < TOLERANCE_SYS_MEM
        assert abs(psutil_result.available - available) < TOLERANCE_SYS_MEM
        assert abs(psutil_result.free - free) < TOLERANCE_SYS_MEM

    def test_swap_memory(self):
        out = sh('/usr/sbin/lsps -a')
        # From the man page, "The size is given in megabytes" so we assume
        # we'll always have 'MB' in the result
        # TODO maybe try to use "swap -l" to check "used" too, but its units
        # are not guaranteed to be "MB" so parsing may not be consistent
        matchobj = re.search(
            r"(?P<space>\S+)\s+"
            r"(?P<vol>\S+)\s+"
            r"(?P<vg>\S+)\s+"
            r"(?P<size>\d+)MB",
            out,
        )

        assert matchobj is not None

        total_mb = int(matchobj.group("size"))
        MB = 1024**2
        psutil_result = psutil.swap_memory()
        # we divide our result by MB instead of multiplying the lsps value by
        # MB because lsps may round down, so we round down too
        assert int(psutil_result.total / MB) == total_mb

    def test_cpu_stats(self):
        out = sh('/usr/bin/mpstat -a')

        re_pattern = r"ALL\s*"
        for field in [
            "min",
            "maj",
            "mpcs",
            "mpcr",
            "dev",
            "soft",
            "dec",
            "ph",
            "cs",
            "ics",
            "bound",
            "rq",
            "push",
            "S3pull",
            "S3grd",
            "S0rd",
            "S1rd",
            "S2rd",
            "S3rd",
            "S4rd",
            "S5rd",
            "sysc",
        ]:
            re_pattern += rf"(?P<{field}>\S+)\s+"
        matchobj = re.search(re_pattern, out)

        assert matchobj is not None

        # numbers are usually in the millions so 1000 is ok for tolerance
        CPU_STATS_TOLERANCE = 1000
        psutil_result = psutil.cpu_stats()
        assert (
            abs(psutil_result.ctx_switches - int(matchobj.group("cs")))
            < CPU_STATS_TOLERANCE
        )
        assert (
            abs(psutil_result.syscalls - int(matchobj.group("sysc")))
            < CPU_STATS_TOLERANCE
        )
        assert (
            abs(psutil_result.interrupts - int(matchobj.group("dev")))
            < CPU_STATS_TOLERANCE
        )
        assert (
            abs(psutil_result.soft_interrupts - int(matchobj.group("soft")))
            < CPU_STATS_TOLERANCE
        )

    def test_cpu_count_logical(self):
        out = sh('/usr/bin/mpstat -a')
        mpstat_lcpu = int(re.search(r"lcpu=(\d+)", out).group(1))
        psutil_lcpu = psutil.cpu_count(logical=True)
        assert mpstat_lcpu == psutil_lcpu

    def test_net_if_addrs_names(self):
        out = sh('/etc/ifconfig -l')
        ifconfig_names = set(out.split())
        psutil_names = set(psutil.net_if_addrs().keys())
        assert ifconfig_names == psutil_names


# <!-- @GENESIS_MODULE_END: test_aix -->
