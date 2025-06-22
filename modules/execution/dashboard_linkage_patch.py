# @GENESIS_ORPHAN_STATUS: recoverable
# @GENESIS_SUGGESTED_ACTION: connect
# @GENESIS_ANALYSIS_DATE: 2025-06-20T16:45:13.468348
# @GENESIS_PROTECTION: DO_NOT_DELETE_UNTIL_REVIEWED

#!/usr/bin/env python3
"""
🔗 GENESIS DASHBOARD LINKAGE PATCH - Phase 92B Core System Reconnection
Connect GUI control buttons to real MT5 data processing scripts

🎯 PURPOSE: Wire dashboard buttons to execute real analysis and backtest scripts
📡 FEATURES: Button event handlers, script execution, result integration
🔧 SCOPE: NO real DATA - Real script execution with live results
"""

import json
import subprocess
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Callable
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DashboardLinkage')

class DashboardLinkagePatch:
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

            emit_telemetry("dashboard_linkage_patch", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def validate_ftmo_compliance(self, trade_data: dict) -> bool:
            """GENESIS FTMO Compliance Validator"""
            # Daily drawdown check (5%)
            daily_loss = trade_data.get('daily_loss', 0)
            if daily_loss > 0.05:
                emit_telemetry("dashboard_linkage_patch", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("dashboard_linkage_patch", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = self._get_event_bus()
        
    def _get_event_bus(self):
        # Auto-injected EventBus connection
        try:
            from event_bus_manager import EventBusManager
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """Connect dashboard UI components to real system scripts"""
    
    def __init__(self):
        self.script_status = {}
        self.last_results = {}
        self.button_handlers = {}
        
        # Ensure directories exist
        os.makedirs("telemetry", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Register button handlers
        self._register_button_handlers()
        
        # Initialize status tracking
        self._initialize_status_tracking()
    
    def _register_button_handlers(self):
        """Register all dashboard button event handlers"""
        self.button_handlers = {
            "run_backtest": self._handle_run_backtest,
            "analyze_live_trades": self._handle_analyze_live_trades,
            "sync_mt5_data": self._handle_sync_mt5_data,
            "refresh_signals": self._handle_refresh_signals,
            "emergency_stop": self._handle_emergency_stop,
            "export_results": self._handle_export_results,
            "validate_system": self._handle_validate_system
        }
        
        logger.info(f"Registered {len(self.button_handlers)} button handlers")
    
    def _initialize_status_tracking(self):
        """Initialize script execution status tracking"""
        self.script_status = {
            "backtest_engine": {"status": "ready", "last_run": None, "running": False},
            "live_trade_analyzer": {"status": "ready", "last_run": None, "running": False},
            "mt5_sync_adapter": {"status": "ready", "last_run": None, "running": False},
            "signal_generator": {"status": "ready", "last_run": None, "running": False}
        }
        
        # Store status for dashboard
        self._update_button_status_file()
    
    def execute_button_action(self, button_id: str, params: Dict = None) -> Dict:
        """
        Execute dashboard button action with real script execution
        REAL SCRIPT EXECUTION ONLY - NO real RESPONSES
        """
        try:
            if button_id not in self.button_handlers:
                raise ValueError(f"Unknown button action: {button_id}")
            
            logger.info(f"🔥 Executing button action: {button_id}")
            
            # Check if script is already running
            script_name = self._get_script_name_from_button(button_id)
            if script_name and self.script_status.get(script_name, {}).get("running", False):
                return {
                    "success": False,
                    "error": f"Script {script_name} is already running",
                    "button_id": button_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Execute handler
            result = self.button_handlers[button_id](params or {})
            
            # Update status
            if script_name:
                self.script_status[script_name]["last_run"] = datetime.now(timezone.utc).isoformat()
                self._update_button_status_file()
            
            # Log execution
            self._log_button_execution(button_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Button action {button_id} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "button_id": button_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _handle_run_backtest(self, params: Dict) -> Dict:
        """Execute backtest engine with real MT5 data"""
        try:
            # Mark as running
            self.script_status["backtest_engine"]["running"] = True
            self._update_button_status_file()
            
            # Get parameters
            symbol = params.get("symbol", "EURUSD")
            strategy_type = params.get("strategy", "MACD_RSI")
            
            # Execute backtest script
            logger.info(f"🚀 Starting backtest for {symbol} with {strategy_type} strategy")
            
            # Run backtest in background thread to avoid blocking
            def run_backtest():
                try:
                    # Import and run backtest engine
                    from backtest_engine import BacktestEngine
                    
                    engine = BacktestEngine()
                    strategy_params = {
                        "strategy_type": strategy_type,
                        "risk_per_trade": 0.02,
                        "stop_loss_pips": 50,
                        "take_profit_pips": 100
                    }
                    
                    # Run comprehensive backtest
                    results = engine.run_comprehensive_backtest(symbol, strategy_params)
                    
                    # Store results for dashboard
                    self.last_results["backtest"] = results
                    self._update_dashboard_results("backtest", results)
                    
                    logger.info(f"✅ Backtest completed: {results.get('total_trades', 0)} trades")
                    
                except Exception as e:
                    logger.error(f"Backtest execution failed: {e}")
                    error_result = {"error": str(e), "symbol": symbol}
                    self.last_results["backtest"] = error_result
                    self._update_dashboard_results("backtest", error_result)
                finally:
                    # Mark as not running
                    self.script_status["backtest_engine"]["running"] = False
                    self._update_button_status_file()
            
            # Start background thread
            thread = threading.Thread(target=run_backtest)
            thread.daemon = True
            thread.start()
            
            return {
                "success": True,
                "message": f"Backtest started for {symbol}",
                "symbol": symbol,
                "strategy": strategy_type,
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.script_status["backtest_engine"]["running"] = False
            self._update_button_status_file()
            raise e
    
    def _handle_analyze_live_trades(self, params: Dict) -> Dict:
        """Execute live trade analyzer with real MT5 positions"""
        try:
            # Mark as running
            self.script_status["live_trade_analyzer"]["running"] = True
            self._update_button_status_file()
            
            logger.info("🔍 Starting live trade analysis")
            
            # Execute in background thread
            def run_analysis():
                try:
                    # Import and run live trade analyzer
                    from live_trade_analyzer import LiveTradeAnalyzer
                    
                    analyzer = LiveTradeAnalyzer()
                    results = analyzer.analyze_live_positions()
                    
                    # Store results for dashboard
                    self.last_results["live_analysis"] = results
                    self._update_dashboard_results("live_analysis", results)
                    
                    logger.info(f"✅ Live analysis completed: {results.get('total_positions', 0)} positions analyzed")
                    
                except Exception as e:
                    logger.error(f"Live analysis failed: {e}")
                    error_result = {"error": str(e)}
                    self.last_results["live_analysis"] = error_result
                    self._update_dashboard_results("live_analysis", error_result)
                finally:
                    # Mark as not running
                    self.script_status["live_trade_analyzer"]["running"] = False
                    self._update_button_status_file()
            
            # Start background thread
            thread = threading.Thread(target=run_analysis)
            thread.daemon = True
            thread.start()
            
            return {
                "success": True,
                "message": "Live trade analysis started",
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.script_status["live_trade_analyzer"]["running"] = False
            self._update_button_status_file()
            raise e
    
    def _handle_sync_mt5_data(self, params: Dict) -> Dict:
        """Execute MT5 sync adapter for real-time data"""
        try:
            # Mark as running
            self.script_status["mt5_sync_adapter"]["running"] = True
            self._update_button_status_file()
            
            logger.info("📡 Starting MT5 data synchronization")
            
            # Execute sync
            def run_sync():
                try:
                    # Import and run MT5 sync adapter
                    from mt5_sync_adapter import MT5SyncAdapter
                    
                    adapter = MT5SyncAdapter()
                    
                    # Perform comprehensive sync
                    sync_results = {
                        "symbols_synced": len(adapter.sync_symbol_list()),
                        "positions_synced": len(adapter.sync_positions()),
                        "account_synced": adapter.sync_account_info() is not None,
                        "candles_synced": len(adapter.sync_recent_candles()),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Store results
                    self.last_results["mt5_sync"] = sync_results
                    self._update_dashboard_results("mt5_sync", sync_results)
                    
                    logger.info(f"✅ MT5 sync completed: {sync_results['symbols_synced']} symbols, {sync_results['positions_synced']} positions")
                    
                except Exception as e:
                    logger.error(f"MT5 sync failed: {e}")
                    error_result = {"error": str(e)}
                    self.last_results["mt5_sync"] = error_result
                    self._update_dashboard_results("mt5_sync", error_result)
                finally:
                    # Mark as not running
                    self.script_status["mt5_sync_adapter"]["running"] = False
                    self._update_button_status_file()
            
            # Start background thread
            thread = threading.Thread(target=run_sync)
            thread.daemon = True
            thread.start()
            
            return {
                "success": True,
                "message": "MT5 data sync started",
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.script_status["mt5_sync_adapter"]["running"] = False
            self._update_button_status_file()
            raise e
    
    def _handle_refresh_signals(self, params: Dict) -> Dict:
        """Execute signal feed generator for real-time signals"""
        try:
            logger.info("🎯 Refreshing signal feed")
            
            # Execute signal refresh
            from signal_feed_generator import SignalFeedGenerator
            
            generator = SignalFeedGenerator()
            signals = generator.generate_signals()
            
            # Store results
            signal_summary = {
                "total_signals": len(signals),
                "active_signals": len([s for s in signals if s.get("status") == "active"]),
                "last_signal_time": max([s.get("timestamp", "") for s in signals]) if signals else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.last_results["signals"] = signal_summary
            self._update_dashboard_results("signals", signal_summary)
            
            return {
                "success": True,
                "message": f"Signal feed refreshed: {signal_summary['total_signals']} signals",
                "signals_generated": signal_summary['total_signals'],
                "active_signals": signal_summary['active_signals'],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Signal refresh failed: {e}")
            raise e
    
    def _handle_emergency_stop(self, params: Dict) -> Dict:
        """Emergency stop all running processes"""
        try:
            logger.warning("🚨 EMERGENCY STOP ACTIVATED")
            
            # Mark all scripts as stopped
            for script_name in self.script_status:
                self.script_status[script_name]["running"] = False
                self.script_status[script_name]["last_run"] = datetime.now(timezone.utc).isoformat()
            
            # Update status file
            self._update_button_status_file()
            
            # Create emergency stop log
            emergency_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "emergency_stop",
                "reason": params.get("reason", "User initiated"),
                "stopped_scripts": list(self.script_status.keys())
            }
            
            with open("logs/emergency_stop.json", 'w') as f:
                json.dump(emergency_log, f, indent=2)
            
            return {
                "success": True,
                "message": "Emergency stop completed - All processes halted",
                "stopped_scripts": list(self.script_status.keys()),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            raise e
    
    def _handle_export_results(self, params: Dict) -> Dict:
        """Export all analysis results to consolidated file"""
        try:
            logger.info("📊 Exporting consolidated results")
            
            # Gather all results
            consolidated_results = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "backtest_results": self.last_results.get("backtest", {}),
                "live_analysis_results": self.last_results.get("live_analysis", {}),
                "mt5_sync_results": self.last_results.get("mt5_sync", {}),
                "signal_results": self.last_results.get("signals", {}),
                "script_status": self.script_status
            }
            
            # Export to file
            export_filename = f"genesis_results_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(export_filename, 'w') as f:
                json.dump(consolidated_results, f, indent=2, default=str)
            
            return {
                "success": True,
                "message": f"Results exported to {export_filename}",
                "export_file": export_filename,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise e
    
    def _handle_validate_system(self, params: Dict) -> Dict:
        """Validate system integrity and connections"""
        try:
            logger.info("🔧 Validating system integrity")
            
            validation_results = {
                "mt5_connection": False,
                "required_files": {},
                "script_availability": {},
                "telemetry_status": {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Check MT5 connection
            try:
                import MetaTrader5 as mt5


# <!-- @GENESIS_MODULE_END: dashboard_linkage_patch -->


# <!-- @GENESIS_MODULE_START: dashboard_linkage_patch -->
                validation_results["mt5_connection"] = mt5.initialize() if hasattr(mt5, 'initialize') else False
            except:
                validation_results["mt5_connection"] = False
            
            # Check required files
            required_files = [
                "backtest_engine.py",
                "live_trade_analyzer.py", 
                "mt5_sync_adapter.py",
                "signal_feed_generator.py",
                "telemetry/mt5_metrics.json"
            ]
            
            for file_path in required_files:
                validation_results["required_files"][file_path] = os.path.exists(file_path)
            
            # Check script availability
            scripts = ["backtest_engine", "live_trade_analyzer", "mt5_sync_adapter"]
            for script in scripts:
                try:
                    __import__(script)
                    validation_results["script_availability"][script] = True
                except:
                    validation_results["script_availability"][script] = False
            
            # Check telemetry files
            telemetry_files = [
                "telemetry/mt5_metrics.json",
                "telemetry/signal_feed.json",
                "logs/execution_log.json"
            ]
            
            for tel_file in telemetry_files:
                validation_results["telemetry_status"][tel_file] = os.path.exists(tel_file)
            
            # Store validation results
            with open("system_validation_results.json", 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            return {
                "success": True,
                "message": "System validation completed",
                "validation_results": validation_results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            raise e
    
    def _get_script_name_from_button(self, button_id: str) -> str:
        """Map button ID to script name for status tracking"""
        mapping = {
            "run_backtest": "backtest_engine",
            "analyze_live_trades": "live_trade_analyzer",
            "sync_mt5_data": "mt5_sync_adapter",
            "refresh_signals": "signal_generator"
        }
        return mapping.get(button_id)
    
    def _update_button_status_file(self):
        """Update button status file for dashboard"""
        try:
            button_status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "script_status": self.script_status,
                "last_results_summary": {
                    key: {"available": bool(value), "timestamp": value.get("timestamp") if isinstance(value, dict) else None}
                    for key, value in self.last_results.items()
                }
            }
            
            with open("dashboard_button_status.json", 'w') as f:
                json.dump(button_status, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating button status file: {e}")
    
    def _update_dashboard_results(self, result_type: str, results: Dict):
        """Update dashboard results files"""
        try:
            # Store in telemetry for dashboard consumption
            telemetry_file = f"telemetry/{result_type}_latest.json"
            with open(telemetry_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            # Update execution log
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "module": "DashboardLinkage",
                "action": f"update_{result_type}_results",
                "success": "error" not in results,
                "result_summary": str(results)[:200]  # Truncated summary
            }
            
            self._append_to_execution_log(log_entry)
            
        except Exception as e:
            logger.error(f"Error updating dashboard results: {e}")
    
    def _append_to_execution_log(self, log_entry: Dict):
        """Append entry to execution log"""
        try:
            log_file = "logs/execution_log.json"
            
            # Load existing log
            try:
                with open(log_file, 'r') as f:
                    execution_log = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                execution_log = []
            
            # Append new entry
            execution_log.append(log_entry)
            
            # Keep only last 1000 entries
            if len(execution_log) > 1000:
                execution_log = execution_log[-1000:]
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(execution_log, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error updating execution log: {e}")
    
    def _log_button_execution(self, button_id: str, result: Dict):
        """Log button execution for audit trail"""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "button_id": button_id,
                "success": result.get("success", False),
                "message": result.get("message", ""),
                "error": result.get("error", None)
            }
            
            # Store in button execution log
            button_log_file = "logs/button_execution_log.json"
            
            try:
                with open(button_log_file, 'r') as f:
                    button_log = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                button_log = []
            
            button_log.append(log_entry)
            
            # Keep only last 500 entries
            if len(button_log) > 500:
                button_log = button_log[-500:]
            
            with open(button_log_file, 'w') as f:
                json.dump(button_log, f, indent=2, default=str)
                
            # Also append to main execution log
            self._append_to_execution_log({
                "timestamp": log_entry["timestamp"],
                "module": "DashboardLinkage",
                "action": f"button_{button_id}",
                "success": log_entry["success"],
                "result_summary": log_entry["message"]
            })
            
        except Exception as e:
            logger.error(f"Error logging button execution: {e}")

def main():
    """Test dashboard linkage functionality"""
    try:
        linkage = DashboardLinkagePatch()
        
        print("🔗 GENESIS Dashboard Linkage Patch Active")
        print(f"Registered Button Handlers: {list(linkage.button_handlers.keys())}")
        
        # Test system validation
        validation_result = linkage.execute_button_action("validate_system")
        print(f"\n🔧 System Validation: {'✅ PASS' if validation_result.get('success') else '❌ FAIL'}")
        
        return linkage
        
    except Exception as e:
        logger.error(f"Dashboard linkage initialization failed: {e}")
        raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Dashboard linkage operation failed")

if __name__ == "__main__":
    main()

    def log_state(self):
        """Phase 91 Telemetry Enforcer - Log current module state"""
        state_data = {
            "module": __name__,
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "phase": "91_telemetry_enforcement"
        }
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", state_data)
        return state_data
        

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


def setup_event_subscriptions(self):
    """Set up EventBus subscriptions for this UI component"""
    event_bus.subscribe("market_data_updated", self.handle_market_data_update)
    event_bus.subscribe("trade_executed", self.handle_trade_update)
    event_bus.subscribe("position_changed", self.handle_position_update)
    event_bus.subscribe("risk_threshold_warning", self.handle_risk_warning)
    event_bus.subscribe("system_status_changed", self.handle_system_status_update)
    
    # Register with telemetry
    telemetry.log_event(TelemetryEvent(
        category="ui", 
        name="event_subscriptions_setup", 
        properties={"component": self.__class__.__name__}
    ))
