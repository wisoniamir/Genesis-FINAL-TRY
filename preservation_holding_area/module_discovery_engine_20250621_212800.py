
# 📊 GENESIS Telemetry Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.telemetry import emit_telemetry, TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    def emit_telemetry(module, event, data): 
        print(f"TELEMETRY: {module}.{event} - {data}")
    class TelemetryManager:
        def emergency_stop(self, reason: str = "Manual trigger") -> bool:
                """GENESIS Emergency Kill Switch"""
                try:
                    # Emit emergency event
                    if hasattr(self, 'event_bus') and self.event_bus:
                        emit_event("emergency_stop", {
                            "module": "module_discovery_engine",
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
                    print(f"Emergency stop error in module_discovery_engine: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "module_discovery_engine",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("module_discovery_engine", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in module_discovery_engine: {e}")
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


# -*- coding: utf-8 -*-
# <!-- @GENESIS_MODULE_START: core_module_discovery_engine -->

"""
🔍 GENESIS MODULE DISCOVERY ENGINE v7.0.0
DYNAMIC MODULE DISCOVERY & SYNC SYSTEM

🚨 ARCHITECT MODE v7.0.0 - DISCOVERY PATTERNS:
- NO HARDENED IMPLEMENTATIONS: Dynamic discovery only
- REAL-TIME MODULE DISCOVERY: Live scanning and detection
- INTER-MODULE SYNC: Auto-discovery of module dependencies
- MT5 DISCOVERY: Dynamic MT5 capability detection
- EVENTBUS DISCOVERY: Auto-discovery of event routing
"""

import os
import sys
import time
import threading
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging
from dataclasses import dataclass, field
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModuleDiscoveryInfo:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "module_discovery_engine",
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
                print(f"Emergency stop error in module_discovery_engine: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "module_discovery_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("module_discovery_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in module_discovery_engine: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "module_discovery_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in module_discovery_engine: {e}")
    """Information about discovered module"""
    name: str
    path: str
    module_type: str
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    event_emitters: List[str] = field(default_factory=list)
    event_listeners: List[str] = field(default_factory=list)
    mt5_required: bool = False
    last_discovered: datetime = field(default_factory=datetime.now)
    status: str = "discovered"
    health_score: float = 1.0

@dataclass
class MT5DiscoveryResult:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "module_discovery_engine",
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
                print(f"Emergency stop error in module_discovery_engine: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "module_discovery_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("module_discovery_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in module_discovery_engine: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "module_discovery_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in module_discovery_engine: {e}")
    """MT5 discovery results"""
    available: bool = False
    connection_status: str = "unknown"
    account_info: Dict[str, Any] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    terminal_info: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    discovery_timestamp: datetime = field(default_factory=datetime.now)

class GenesisModuleDiscoveryEngine:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "module_discovery_engine",
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
                print(f"Emergency stop error in module_discovery_engine: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "module_discovery_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("module_discovery_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in module_discovery_engine: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "module_discovery_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in module_discovery_engine: {e}")
    """
    🔍 GENESIS MODULE DISCOVERY ENGINE
    
    Dynamically discovers, maps, and synchronizes all GENESIS modules
    without hardcoded implementations. Uses discovery patterns for:
    - Module capability detection
    - Dependency mapping
    - Event bus routing discovery
    - MT5 integration discovery
    - Real-time health monitoring
    """
    
    def __init__(self):
        self.discovered_modules: Dict[str, ModuleDiscoveryInfo] = {}
        self.mt5_discovery: Optional[MT5DiscoveryResult] = None
        self.event_bus_discovery: Dict[str, Any] = {}
        self.sync_threads: List[threading.Thread] = []
        self.discovery_active = False
        self.discovery_interval = 30.0  # seconds
        
        # Discovery paths
        self.discovery_paths = [
            'core', 'modules', 'connectors', 'execution', 
            'data', 'compliance', 'backtest', 'interface',
            'brokers', 'strategies', 'signals', 'alerts'
        ]
        
        # Module type patterns
        self.module_type_patterns = {
            'connector': ['mt5', 'telegram', 'news', 'api'],
            'strategy': ['sniper', 'scalp', 'swing', 'pattern'],
            'execution': ['order', 'risk', 'kill', 'stop'],
            'signal': ['scanner', 'detector', 'analyzer'],
            'compliance': ['ftmo', 'risk', 'audit', 'monitor'],
            'data': ['feed', 'store', 'cache', 'history'],
            'interface': ['dashboard', 'gui', 'web', 'api']
        }
        
        logger.info("🔍 GENESIS Module Discovery Engine initialized")
    
    async def start_discovery(self):
        """Start the discovery process"""
        self.discovery_active = True
        logger.info("🔍 Starting GENESIS module discovery...")
        
        # Initial comprehensive discovery
        await self.comprehensive_discovery()
        
        # Start continuous discovery thread
        discovery_thread = threading.Thread(
            target=self._continuous_discovery_loop,
            daemon=True
        )
        discovery_thread.start()
        self.sync_threads.append(discovery_thread)
        
        logger.info("🔍 Discovery engine started successfully")
    
    async def comprehensive_discovery(self):
        """Perform comprehensive module discovery"""
        logger.info("🔍 Starting comprehensive discovery scan...")
        
        # Discover modules in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            discovery_tasks = [
                executor.submit(self._discover_modules_in_path, path)
                for path in self.discovery_paths
            ]
            
            # Wait for all discovery tasks
            for task in discovery_tasks:
                try:
                    task.result(timeout=30)
                except Exception as e:
                    logger.error(f"Discovery task failed: {e}")
        
        # Discover MT5 capabilities
        await self._discover_mt5_capabilities()
        
        # Discover event bus routing
        await self._discover_event_bus_routing()
        
        # Map inter-module dependencies
        await self._map_module_dependencies()
        
        # Validate discovery results
        await self._validate_discovery_results()
        
        logger.info(f"🔍 Discovery complete: {len(self.discovered_modules)} modules found")
    
    def _discover_modules_in_path(self, base_path: str):
        """Discover modules in a specific path"""
        if not os.path.exists(base_path):
            return
            
        try:
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        file_path = os.path.join(root, file)
                        self._analyze_module_file(file_path, base_path)
                        
        except Exception as e:
            logger.error(f"Error discovering modules in {base_path}: {e}")
    
    def _analyze_module_file(self, file_path: str, base_path: str):
        """Analyze a Python file for module capabilities"""
        try:
            # Generate module name
            relative_path = os.path.relpath(file_path, base_path)
            module_name = relative_path.replace(os.sep, '.').replace('.py', '')
            
            # Read file content for analysis
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Discover module capabilities
            capabilities = self._extract_capabilities(content)
            dependencies = self._extract_dependencies(content)
            event_emitters = self._extract_event_emitters(content)
            event_listeners = self._extract_event_listeners(content)
            mt5_required = self._check_mt5_dependency(content)
            module_type = self._determine_module_type(module_name, content)
            
            # Create discovery info
            discovery_info = ModuleDiscoveryInfo(
                name=module_name,
                path=file_path,
                module_type=module_type,
                capabilities=capabilities,
                dependencies=dependencies,
                event_emitters=event_emitters,
                event_listeners=event_listeners,
                mt5_required=mt5_required
            )
            
            self.discovered_modules[module_name] = discovery_info
            logger.debug(f"✅ Discovered module: {module_name} ({module_type})")
            
        except Exception as e:
            logger.warning(f"Could not analyze {file_path}: {e}")
    
    def _extract_capabilities(self, content: str) -> List[str]:
        """Extract module capabilities from source code"""
        capabilities = []
        
        # Look for function definitions that indicate capabilities
        capability_patterns = [
            ('trading', ['def place_order', 'def execute_trade', 'def close_position']),
            ('analysis', ['def analyze', 'def calculate', 'def detect']),
            ('monitoring', ['def monitor', 'def check', 'def watch']),
            ('data_processing', ['def process_data', 'def transform', 'def filter']),
            ('event_handling', ['def emit_event', 'def handle_event', 'def subscribe']),
            ('mt5_integration', ['mt5.', 'MetaTrader5', 'terminal_info']),
            ('risk_management', ['def check_risk', 'def validate_risk', 'stop_loss']),
            ('signal_generation', ['def generate_signal', 'def detect_pattern', 'confluence']),
            ('compliance', ['def audit', 'def validate_compliance', 'ftmo']),
            ('dashboard', ['QWidget', 'QMainWindow', 'streamlit'])
        ]
        
        for capability_name, patterns in capability_patterns:
            if any(pattern in content for pattern in patterns):
                capabilities.append(capability_name)
        
        return capabilities
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract module dependencies from imports"""
        dependencies = []
        
        # Extract import statements
        import_lines = [line.strip() for line in content.split('\n') 
                       if line.strip().startswith(('import ', 'from '))]
        
        for line in import_lines:
            if 'MetaTrader5' in line or 'mt5' in line:
                dependencies.append('mt5')
            elif 'PyQt5' in line or 'tkinter' in line:
                dependencies.append('gui')
            elif 'streamlit' in line:
                dependencies.append('web_interface')
            elif 'redis' in line:
                dependencies.append('cache')
            elif 'requests' in line:
                dependencies.append('http')
            elif 'websocket' in line:
                dependencies.append('websocket')
            elif 'telegram' in line:
                dependencies.append('telegram')
            elif any(pattern in line for pattern in ['event_bus', 'EventBus']):
                dependencies.append('event_bus')
        
        return list(set(dependencies))
    
    def _extract_event_emitters(self, content: str) -> List[str]:
        """Extract event emitters from source code"""
        emitters = []
        
        # Look for event emission patterns
        emit_patterns = [
            'emit_event(',
            'publish(',
            'send_signal(',
            'trigger_event(',
            'broadcast('
        ]
        
        for pattern in emit_patterns:
            if pattern in content:
                # Try to extract event names
                lines = content.split('\n')
                for line in lines:
                    if pattern in line:
                        # Simple extraction of event names
                        if '"' in line:
                            event_name = line.split('"')[1]
                            emitters.append(event_name)
                        elif "'" in line:
                            event_name = line.split("'")[1]
                            emitters.append(event_name)
        
        return list(set(emitters))
    
    def _extract_event_listeners(self, content: str) -> List[str]:
        """Extract event listeners from source code"""
        listeners = []
        
        # Look for event listening patterns
        listen_patterns = [
            'subscribe_to_event(',
            'listen_for(',
            'on_event(',
            'handle_signal(',
            'register_callback('
        ]
        
        for pattern in listen_patterns:
            if pattern in content:
                # Extract event names being listened to
                lines = content.split('\n')
                for line in lines:
                    if pattern in line:
                        if '"' in line:
                            event_name = line.split('"')[1]
                            listeners.append(event_name)
                        elif "'" in line:
                            event_name = line.split("'")[1]
                            listeners.append(event_name)
        
        return list(set(listeners))
    
    def _check_mt5_dependency(self, content: str) -> bool:
        """Check if module requires MT5"""
        mt5_indicators = [
            'MetaTrader5',
            'mt5.',
            'terminal_info',
            'account_info',
            'positions_get',
            'orders_get',
            'copy_rates'
        ]
        
        return any(indicator in content for indicator in mt5_indicators)
    
    def _determine_module_type(self, module_name: str, content: str) -> str:
        """Determine module type based on name and content"""
        module_name_lower = module_name.lower()
        content_lower = content.lower()
        
        for module_type, patterns in self.module_type_patterns.items():
            if any(pattern in module_name_lower or pattern in content_lower 
                   for pattern in patterns):
                return module_type
        
        return 'utility'
    
    async def _discover_mt5_capabilities(self):
        """Discover MT5 capabilities and connection status"""
        logger.info("🔍 Discovering MT5 capabilities...")
        
        try:
            # Try to import MT5
            import MetaTrader5 as mt5
            
            # Attempt connection
            if mt5.initialize():
                # Get account info
                account_info = mt5.account_info()
                account_dict = account_info._asdict() if account_info else {}
                
                # Get terminal info
                terminal_info = mt5.terminal_info()
                terminal_dict = terminal_info._asdict() if terminal_info else {}
                
                # Get available symbols
                symbols = mt5.symbols_get()
                symbol_names = [s.name for s in symbols] if symbols else []
                
                # Determine capabilities
                capabilities = ['trading', 'data_feed', 'account_management']
                if len(symbol_names) > 0:
                    capabilities.append('symbol_discovery')
                if account_dict.get('trade_allowed', False):
                    capabilities.append('trade_execution')
                
                self.mt5_discovery = MT5DiscoveryResult(
                    available=True,
                    connection_status='connected',
                    account_info=account_dict,
                    symbols=symbol_names,
                    terminal_info=terminal_dict,
                    capabilities=capabilities
                )
                
                logger.info(f"✅ MT5 discovered: {len(symbol_names)} symbols, account {account_dict.get('login', 'unknown')}")
                
            else:
                self.mt5_discovery = MT5DiscoveryResult(
                    available=True,
                    connection_status='connection_failed'
                )
                logger.warning("⚠️ MT5 available but connection failed")
                
        except ImportError:
            self.mt5_discovery = MT5DiscoveryResult(
                available=False,
                connection_status='not_installed'
            )
            logger.warning("⚠️ MT5 not installed")
        except Exception as e:
            self.mt5_discovery = MT5DiscoveryResult(
                available=False,
                connection_status=f'error: {str(e)}'
            )
            logger.error(f"MT5 discovery error: {e}")
    
    async def _discover_event_bus_routing(self):
        """Discover event bus routing capabilities"""
        logger.info("🔍 Discovering event bus routing...")
        
        # Look for event bus modules
        event_bus_modules = [
            module for module in self.discovered_modules.values()
            if 'event_handling' in module.capabilities
        ]
        
        self.event_bus_discovery = {
            'available_modules': [m.name for m in event_bus_modules],
            'total_emitters': sum(len(m.event_emitters) for m in self.discovered_modules.values()),
            'total_listeners': sum(len(m.event_listeners) for m in self.discovered_modules.values()),
            'event_types': list(set(
                event for module in self.discovered_modules.values()
                for event in module.event_emitters + module.event_listeners
            ))
        }
        
        logger.info(f"✅ Event bus discovery: {len(event_bus_modules)} modules, {len(self.event_bus_discovery['event_types'])} event types")
    
    async def _map_module_dependencies(self):
        """Map inter-module dependencies"""
        logger.info("🔍 Mapping module dependencies...")
        
        dependency_graph = {}
        
        for module_name, module_info in self.discovered_modules.items():
            dependencies = []
            
            # Check for direct dependencies
            for other_name, other_info in self.discovered_modules.items():
                if other_name != module_name:
                    # Check if module emits events that other module listens to
                    if any(event in other_info.event_listeners for event in module_info.event_emitters):
                        dependencies.append(other_name)
                    
                    # Check if module listens to events that other module emits
                    if any(event in module_info.event_listeners for event in other_info.event_emitters):
                        dependencies.append(other_name)
            
            dependency_graph[module_name] = list(set(dependencies))
        
        # Update module info with discovered dependencies
        for module_name, deps in dependency_graph.items():
            if module_name in self.discovered_modules:
                self.discovered_modules[module_name].dependencies.extend(deps)
                self.discovered_modules[module_name].dependencies = list(set(
                    self.discovered_modules[module_name].dependencies
                ))
        
        logger.info("✅ Module dependency mapping complete")
    
    async def _validate_discovery_results(self):
        """Validate discovery results"""
        logger.info("🔍 Validating discovery results...")
        
        validation_results = {
            'total_modules': len(self.discovered_modules),
            'mt5_modules': len([m for m in self.discovered_modules.values() if m.mt5_required]),
            'trading_modules': len([m for m in self.discovered_modules.values() if 'trading' in m.capabilities]),
            'interface_modules': len([m for m in self.discovered_modules.values() if m.module_type == 'interface']),
            'event_connectivity': len(self.event_bus_discovery.get('event_types', [])),
            'mt5_available': self.mt5_discovery.available if self.mt5_discovery else False
        }
        
        logger.info(f"✅ Discovery validation: {validation_results}")
        
        # Save discovery results
        await self._save_discovery_results(validation_results)
    
    async def _save_discovery_results(self, validation_results: Dict[str, Any]):
        """Save discovery results to files"""
        try:
            # Save to module registry
            registry_data = {
                'discovery_timestamp': datetime.now().isoformat(),
                'validation_results': validation_results,
                'modules': {
                    name: {
                        'name': info.name,
                        'path': info.path,
                        'type': info.module_type,
                        'capabilities': info.capabilities,
                        'dependencies': info.dependencies,
                        'mt5_required': info.mt5_required,
                        'status': info.status
                    }
                    for name, info in self.discovered_modules.items()
                },
                'mt5_discovery': {
                    'available': self.mt5_discovery.available,
                    'connection_status': self.mt5_discovery.connection_status,
                    'capabilities': self.mt5_discovery.capabilities,
                    'symbol_count': len(self.mt5_discovery.symbols)
                } if self.mt5_discovery else None,
                'event_bus_discovery': self.event_bus_discovery
            }
            
            # Write to module_registry.json
            with open('module_registry.json', 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info("✅ Discovery results saved to module_registry.json")
            
        except Exception as e:
            logger.error(f"Failed to save discovery results: {e}")
    
    def _continuous_discovery_loop(self):
        """Continuous discovery loop"""
        while self.discovery_active:
            try:
                time.sleep(self.discovery_interval)
                
                # Run lightweight discovery update
                asyncio.run(self._lightweight_discovery_update())
                
            except Exception as e:
                logger.error(f"Continuous discovery error: {e}")
    
    async def _lightweight_discovery_update(self):
        """Lightweight discovery update"""
        logger.debug("🔍 Running lightweight discovery update...")
        
        # Check module health
        for module_name, module_info in self.discovered_modules.items():
            try:
                # Simple health check based on file modification
                if os.path.exists(module_info.path):
                    mtime = os.path.getmtime(module_info.path)
                    if mtime > module_info.last_discovered.timestamp():
                        module_info.last_discovered = datetime.now()
                        module_info.health_score = 1.0
                        logger.debug(f"Module {module_name} updated")
                else:
                    module_info.health_score = 0.0
                    module_info.status = "missing"
                    
            except Exception as e:
                logger.warning(f"Health check failed for {module_name}: {e}")
                module_info.health_score = max(0.0, module_info.health_score - 0.1)
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get discovery summary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_modules': len(self.discovered_modules),
            'module_types': {
                module_type: len([m for m in self.discovered_modules.values() if m.module_type == module_type])
                for module_type in set(m.module_type for m in self.discovered_modules.values())
            },
            'mt5_discovery': {
                'available': self.mt5_discovery.available if self.mt5_discovery else False,
                'status': self.mt5_discovery.connection_status if self.mt5_discovery else 'unknown',
                'symbols_count': len(self.mt5_discovery.symbols) if self.mt5_discovery else 0
            },
            'event_bus': self.event_bus_discovery,
            'healthy_modules': len([m for m in self.discovered_modules.values() if m.health_score > 0.8])
        }
    
    def get_module_by_capability(self, capability: str) -> List[ModuleDiscoveryInfo]:
        """Get modules by capability"""
        return [
            module for module in self.discovered_modules.values()
            if capability in module.capabilities
        ]
    
    def get_trading_modules(self) -> List[ModuleDiscoveryInfo]:
        """Get all trading-capable modules"""
        return self.get_module_by_capability('trading')
    
    def get_mt5_modules(self) -> List[ModuleDiscoveryInfo]:
        """Get all MT5-dependent modules"""
        return [
            module for module in self.discovered_modules.values()
            if module.mt5_required
        ]
    
    def stop_discovery(self):
        """Stop discovery engine"""
        self.discovery_active = False
        logger.info("🔍 Discovery engine stopped")

# Global discovery engine instance
discovery_engine = GenesisModuleDiscoveryEngine()

# Convenience functions
async def start_discovery():
    """Start the discovery engine"""
    await discovery_engine.start_discovery()

def get_discovery_summary():
    """Get discovery summary"""
    return discovery_engine.get_discovery_summary()

def get_trading_modules():
    """Get trading modules"""
    return discovery_engine.get_trading_modules()

def get_mt5_status():
    """Get MT5 status"""
    return discovery_engine.mt5_discovery

def get_discovered_modules():
    """Get all discovered modules"""
    return discovery_engine.discovered_modules

# <!-- @GENESIS_MODULE_END: core_module_discovery_engine -->


def check_ftmo_limits(order_volume: float, symbol: str) -> bool:
    """Check order against FTMO trading limits"""
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        logging.error("Failed to get account info")
        return False
    
    # Calculate position size as percentage of account
    equity = account_info.equity
    max_risk_percent = 0.05  # 5% max risk per trade (FTMO rule)
    
    # Calculate potential loss
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}")
        return False
    
    # Check if order volume exceeds max risk
    if (order_volume * symbol_info.trade_tick_value) > (equity * max_risk_percent):
        logging.warning(f"Order volume {order_volume} exceeds FTMO risk limit of {equity * max_risk_percent}")
        return False
    
    # Check daily loss limit
    daily_loss_limit = equity * 0.05  # 5% daily loss limit
    
    # Get today's closed positions
    from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    positions = mt5.history_deals_get(from_date, datetime.now())
    
    daily_pnl = sum([deal.profit for deal in positions if deal.profit < 0])
    
    if abs(daily_pnl) + (order_volume * symbol_info.trade_tick_value) > daily_loss_limit:
        logging.warning(f"Order would breach FTMO daily loss limit. Current loss: {abs(daily_pnl)}")
        return False
    
    return True


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


def monitor_drawdown(max_drawdown_percent: float = 5.0, daily_limit_percent: float = 5.0) -> Dict:
    """
    Monitor account drawdown against FTMO limits
    
    Args:
        max_drawdown_percent: Maximum allowed drawdown percentage
        daily_limit_percent: Maximum allowed daily loss percentage
        
    Returns:
        Dictionary with drawdown status information
    """
    try:
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info")
            return {"status": "error", "message": "Failed to get account info"}
        
        # Calculate current drawdown
        balance = account_info.balance
        equity = account_info.equity
        
        current_drawdown = (balance - equity) / balance * 100 if balance > 0 else 0
        
        # Get daily high balance
        from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        positions = mt5.history_deals_get(from_date, datetime.now())
        
        daily_starting_balance = balance - sum([deal.profit for deal in positions])
        daily_loss_percent = (daily_starting_balance - equity) / daily_starting_balance * 100 if daily_starting_balance > 0 else 0
        
        # Prepare result
        result = {
            "status": "ok",
            "current_drawdown_percent": current_drawdown,
            "max_drawdown_percent": max_drawdown_percent,
            "drawdown_level": current_drawdown / max_drawdown_percent,  # 0.0 to 1.0+
            "daily_loss_percent": daily_loss_percent,
            "daily_limit_percent": daily_limit_percent,
            "daily_loss_level": daily_loss_percent / daily_limit_percent,  # 0.0 to 1.0+
            "warnings": []
        }
        
        # Check drawdown thresholds
        if current_drawdown > max_drawdown_percent * 0.7:
            result["warnings"].append(f"Drawdown at {current_drawdown:.2f}% approaching maximum of {max_drawdown_percent:.2f}%")
            result["status"] = "warning"
            
        if current_drawdown > max_drawdown_percent:
            result["warnings"].append(f"CRITICAL: Drawdown of {current_drawdown:.2f}% exceeds maximum of {max_drawdown_percent:.2f}%")
            result["status"] = "critical"
            
        # Check daily loss thresholds
        if daily_loss_percent > daily_limit_percent * 0.7:
            result["warnings"].append(f"Daily loss at {daily_loss_percent:.2f}% approaching limit of {daily_limit_percent:.2f}%")
            result["status"] = "warning"
            
        if daily_loss_percent > daily_limit_percent:
            result["warnings"].append(f"CRITICAL: Daily loss of {daily_loss_percent:.2f}% exceeds limit of {daily_limit_percent:.2f}%")
            result["status"] = "critical"
        
        # Emit events for warnings
        if result["status"] in ["warning", "critical"]:
            emit_event("risk_threshold_warning", {
                "status": result["status"],
                "warnings": result["warnings"],
                "timestamp": datetime.now().isoformat()
            })
            
        return result
        
    except Exception as e:
        logging.error(f"Error monitoring drawdown: {str(e)}")
        return {"status": "error", "message": str(e)}


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
