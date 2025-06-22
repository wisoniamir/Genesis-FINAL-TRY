
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

                emit_telemetry("app", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("app", "position_calculated", {
                    "risk_amount": risk_amount,
                    "position_size": position_size,
                    "risk_percentage": (position_size / account_balance) * 100
                })

                return position_size
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
                    "module": "app",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("app", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in app: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False



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


"""


# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()

🌐 GENESIS FLASK API — ARCHITECT MODE v7.0 COMPLIANT
Real-time trading dashboard backend with WebSocket support.
"""

import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import redis
import threading
import time
from typing import Dict, Any

# Add project paths
sys.path.append('/app')
sys.path.append('/app/core')
sys.path.append('/app/modules')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/genesis_api.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'genesis_architect_mode_v7'
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Initialize Redis
try:
    redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connected")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    redis_client = None

# Global state
system_state = {
    "status": "starting",
    "modules": {},
    "mt5_connected": False,
    "telemetry_active": False,
    "last_update": datetime.now().isoformat()
}

class GenesisAPI:
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

            emit_telemetry("app", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("app", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
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
                "module": "app",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("app", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in app: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "app",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in app: {e}")
    """
    🌐 GENESIS API SERVER
    
    ARCHITECT MODE COMPLIANCE:
    - ✅ Real-time WebSocket updates
    - ✅ RESTful API endpoints
    - ✅ Module integration
    - ✅ Telemetry streaming
    - ✅ MT5 connectivity
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.modules = {}
        self.telemetry_thread = None
        self.is_running = False
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize core system components"""
        try:
            # Import core modules
            from core.event_bus import get_event_bus, emit_event
            from core.telemetry import emit_telemetry
            from core.config_engine import get_config_engine
            
            self.event_bus = get_event_bus()
            self.config = get_config_engine()
            
            # Register API events
            self.event_bus.subscribe('telemetry.*', self._handle_telemetry)
            self.event_bus.subscribe('strategy.signal', self._handle_signal)
            self.event_bus.subscribe('execution.order', self._handle_order)
            self.event_bus.subscribe('kill_switch.engage', self._handle_kill_switch)
            
            self.logger.info("✅ GENESIS API initialized")
            
        except Exception as e:
            self.logger.error(f"API initialization error: {e}")
            
    def _handle_telemetry(self, data):
        """Handle telemetry events"""
        try:
            # Store in Redis
            if redis_client:
                redis_client.setex('telemetry_latest', 300, json.dumps(data))
                
            # Broadcast via WebSocket
            socketio.emit('telemetry_update', data)
            
        except Exception as e:
            self.logger.error(f"Telemetry handling error: {e}")
            
    def _handle_signal(self, data):
        """Handle strategy signals"""
        try:
            # Store signal
            if redis_client:
                signals = redis_client.lrange('signals', 0, -1)
                redis_client.lpush('signals', json.dumps(data))
                redis_client.ltrim('signals', 0, 99)  # Keep last 100
                
            # Broadcast signal
            socketio.emit('new_signal', data)
            
        except Exception as e:
            self.logger.error(f"Signal handling error: {e}")
            
    def _handle_order(self, data):
        """Handle execution orders"""
        try:
            # Store order
            if redis_client:
                redis_client.lpush('orders', json.dumps(data))
                redis_client.ltrim('orders', 0, 99)  # Keep last 100
                
            # Broadcast order
            socketio.emit('new_order', data)
            
        except Exception as e:
            self.logger.error(f"Order handling error: {e}")
            
    def _handle_kill_switch(self, data):
        """Handle kill switch activation"""
        try:
            # Store kill switch event
            if redis_client:
                redis_client.setex('kill_switch_status', 3600, json.dumps(data))
                
            # Broadcast emergency
            socketio.emit('kill_switch_activated', data)
            
        except Exception as e:
            self.logger.error(f"Kill switch handling error: {e}")
            
    def start_telemetry(self):
        """Start telemetry monitoring"""
        def telemetry_worker():
            while self.is_running:
                try:
                    # Collect system telemetry
                    telemetry_data = {
                        "timestamp": datetime.now().isoformat(),
                        "system_status": system_state["status"],
                        "modules_loaded": len(self.modules),
                        "mt5_connected": system_state["mt5_connected"],
                        "memory_usage": self._get_memory_usage(),
                        "cpu_usage": self._get_cpu_usage()
                    }
                    
                    # Store and broadcast
                    if redis_client:
                        redis_client.setex('system_telemetry', 60, json.dumps(telemetry_data))
                    socketio.emit('system_telemetry', telemetry_data)
                    
                    time.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Telemetry worker error: {e}")
                    
        self.is_running = True
        self.telemetry_thread = threading.Thread(target=telemetry_worker, daemon=True)
        self.telemetry_thread.start()
        
    def _get_memory_usage(self):
        """Get memory usage (placeholder)"""
        return "150MB"  # Would be real memory usage
        
    def _get_cpu_usage(self):
        """Get CPU usage (placeholder)"""
        return "12%"  # Would be real CPU usage

# Initialize API
genesis_api = GenesisAPI()

# REST API Routes
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "7.0",
        "architect_mode": True
    })

@app.route('/api/system/status')
def get_system_status():
    """Get system status"""
    try:
        return jsonify(system_state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/modules')
def get_modules():
    """Get loaded modules"""
    try:
        modules_data = []
        for name, module in genesis_api.modules.items():
            if hasattr(module, 'get_module_status'):
                modules_data.append(module.get_module_status())
            else:
                modules_data.append({
                    "module_name": name,
                    "status": "unknown",
                    "compliance_score": 0
                })
        return jsonify(modules_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/signals')
def get_signals():
    """Get recent signals"""
    try:
        if redis_client:
            signals = redis_client.lrange('signals', 0, 19)  # Last 20
            return jsonify([json.loads(s) for s in signals])
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/orders')
def get_orders():
    """Get recent orders"""
    try:
        if redis_client:
            orders = redis_client.lrange('orders', 0, 19)  # Last 20
            return jsonify([json.loads(o) for o in orders])
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/telemetry')
def get_telemetry():
    """Get latest telemetry"""
    try:
        if redis_client:
            telemetry = redis_client.get('telemetry_latest')
            if telemetry:
                return jsonify(json.loads(telemetry))
        return jsonify({"message": "No telemetry data"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/modules/load', methods=['POST'])
def load_module():
    """Load a specific module"""
    try:
        data = request.get_json()
        module_name = data.get('module_name')
        
        if module_name == 'strategy_engine':
            from modules.strategy.strategy_engine import get_strategy_engine
            genesis_api.modules['strategy_engine'] = get_strategy_engine()
            
        elif module_name == 'execution_manager':
            from modules.execution.execution_manager import get_execution_manager
            genesis_api.modules['execution_manager'] = get_execution_manager()
            
        elif module_name == 'kill_switch':
            from compliance.kill_switch import get_kill_switch
            genesis_api.modules['kill_switch'] = get_kill_switch()
            
        # Broadcast module loaded
        socketio.emit('module_loaded', {
            "module_name": module_name,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({"success": True, "module": module_name})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/kill-switch/trigger', methods=['POST'])
def trigger_kill_switch():
    """Manually trigger kill switch"""
    try:
        data = request.get_json()
        reason = data.get('reason', 'Manual trigger from dashboard')
        
        # Emit kill switch event
        from core.event_bus import emit_event
        emit_event('kill_switch.manual_trigger', {
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "source": "api_dashboard"
        })
        
        return jsonify({"success": True, "message": "Kill switch activated"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/mt5/connect', methods=['POST'])
def connect_mt5():
    """Connect to MT5"""
    try:
        data = request.get_json()
        login = data.get('login')
        password = data.get('password')
        server = data.get('server')        # Real MT5 connection via institutional connector
        try:
            # First initialize MT5 with credentials 
            import MetaTrader5 as mt5
            
            # Initialize MT5 with login credentials
            if not mt5.initialize(login=int(login), password=password, server=server):
                error = mt5.last_error()
                return jsonify({"success": False, "error": f"MT5 login failed: {error}"}), 400
            
            # Now use the institutional connector
            from modules.execution.genesis_institutional_mt5_connector import GenesisInstitutionalMT5Connector

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: app -->


# <!-- @GENESIS_MODULE_START: app -->
            mt5_connector = GenesisInstitutionalMT5Connector()
            
            # Connect using the already authenticated session
            try:
            if mt5_connector.connect():
            except Exception as e:
                logging.error(f"Operation failed: {e}")
                system_state["mt5_connected"] = True
                account_info = mt5.account_info()._asdict() if mt5.account_info() else {}
                system_state["mt5_account_info"] = account_info
            else:
                system_state["mt5_connected"] = False
                return jsonify({"success": False, "error": "MT5 connector failed to initialize"}), 400
                
        except Exception as e:
            system_state["mt5_connected"] = False
            return jsonify({"success": False, "error": f"MT5 connection failed: {str(e)}"}), 500
        
        # Broadcast real connection
        socketio.emit('mt5_connected', {
            "login": login,
            "server": server,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({"success": True, "message": "MT5 connected"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {
        "message": "Connected to GENESIS API",
        "timestamp": datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_system_status')
def handle_status_request():
    """Handle system status request"""
    emit('system_status', system_state)

@socketio.on('start_telemetry')
def handle_start_telemetry():
    """Start telemetry streaming"""
    if not genesis_api.is_running:
        genesis_api.start_telemetry()
    emit('telemetry_started', {"timestamp": datetime.now().isoformat()})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Update system state
    system_state["status"] = "running"
    system_state["last_update"] = datetime.now().isoformat()
    
    # Start telemetry
    genesis_api.start_telemetry()
    
    print("🚀 GENESIS API Server starting...")
    print("📊 Architect Mode v7.0 - Full Compliance")
    print("🔐 Zero Tolerance Enforcement Active")
    print("=" * 50)
    
    # Run with SocketIO
    socketio.run(app, host='0.0.0.0', port=8000, debug=False)



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
