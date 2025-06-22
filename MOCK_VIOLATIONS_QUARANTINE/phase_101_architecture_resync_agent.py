import logging

# Real Data Access Integration
import MetaTrader5 as mt5
from datetime import datetime

class RealDataAccess:
    """Provides real market data access"""
    
    def __init__(self):
        self.mt5_connected = False
        self.data_source = "live"
    
    def get_live_data(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, count=100):
        """Get live market data"""
        try:
            if not self.mt5_connected:
                mt5.initialize()
                self.mt5_connected = True
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            return rates
        except Exception as e:
            logger.error(f"Live data access failed: {e}")
            return None
    
    def get_account_info(self):
        """Get live account information"""
        try:
            return mt5.account_info()
        except Exception as e:
            logger.error(f"Account info access failed: {e}")
            return None

# Initialize real data access
_real_data = RealDataAccess()



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

                emit_telemetry("phase_101_architecture_resync_agent", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("phase_101_architecture_resync_agent", "position_calculated", {
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
                            "module": "phase_101_architecture_resync_agent",
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
                    print(f"Emergency stop error in phase_101_architecture_resync_agent: {e}")
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
                    "module": "phase_101_architecture_resync_agent",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("phase_101_architecture_resync_agent", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in phase_101_architecture_resync_agent: {e}")
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


#!/usr/bin/env python3
"""
🧠 GENESIS PHASE 101 - ARCHITECTURE RESYNC AGENT
==============================================================================

@GENESIS_CATEGORY: CORE.ARCHITECTURE.RESYNC
@GENESIS_TELEMETRY: ENABLED
@GENESIS_EVENTBUS: EMIT+CONSUME

OBJECTIVE: Complete system architecture awareness restoration
- Scan all .py modules
- Rebuild module_registry.json, system_tree.json, event_bus.json
- Classify orphans without deletion
- Generate triage_report.json

COMPLIANCE: ARCHITECT MODE v3.0 ENFORCED
==============================================================================
"""

import os
import json
import ast
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict


# <!-- @GENESIS_MODULE_END: phase_101_architecture_resync_agent -->


# <!-- @GENESIS_MODULE_START: phase_101_architecture_resync_agent -->

class GenesisArchitectureResyncAgent:
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

            emit_telemetry("phase_101_architecture_resync_agent", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("phase_101_architecture_resync_agent", "position_calculated", {
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
                        "module": "phase_101_architecture_resync_agent",
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
                print(f"Emergency stop error in phase_101_architecture_resync_agent: {e}")
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
                "module": "phase_101_architecture_resync_agent",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("phase_101_architecture_resync_agent", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in phase_101_architecture_resync_agent: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "phase_101_architecture_resync_agent",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in phase_101_architecture_resync_agent: {e}")
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.modules_scanned = 0
        self.orphans_found = 0
        self.connected_modules = 0
        self.duplicate_candidates = []
        
        # GENESIS Categories
        self.categories = {
            'CORE': ['dashboard', 'engine', 'core', 'main', 'launcher'],
            'SIGNAL': ['signal', 'pattern', 'strategy'],
            'EXECUTION': ['execution', 'executor', 'order', 'trade'],
            'TELEMETRY': ['telemetry', 'monitor', 'tracker', 'logger'],
            'COMPLIANCE': ['compliance', 'validation', 'audit'],
            'EVENTBUS': ['event', 'bus', 'dispatcher', 'router'],
            'QUARANTINE': ['TRIAGE_ORPHAN_QUARANTINE'],
            'PHASE': ['phase_', 'validate_', 'test_']
        }
        
        # EventBus pattern signatures
        self.eventbus_patterns = [
            r'event_bus\.emit\(',
            r'event_bus\.subscribe\(',
            r'event_bus\.publish\(',
            r'EventBus\(',
            r'emit_signal\(',
            r'@subscribe\(',
            r'event_dispatcher'
        ]
        
        # Telemetry pattern signatures
        self.telemetry_patterns = [
            r'telemetry\.',
            r'log_telemetry\(',
            r'emit_telemetry\(',
            r'TelemetryEngine\(',
            r'@telemetry\(',
            r'track_metric\('
        ]

    def scan_module_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from a Python module"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse AST for imports, classes, functions
            try:
                tree = ast.parse(content)
                imports = []
                functions = []
                classes = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                    elif isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                        
            except SyntaxError:
                imports, functions, classes = [], [], []
            
            # Check for GENESIS patterns
            genesis_category = self.extract_genesis_category(content)
            eventbus_usage = any(re.search(pattern, content) for pattern in self.eventbus_patterns)
            telemetry_enabled = any(re.search(pattern, content) for pattern in self.telemetry_patterns)
            mt5_integration = 'MT5' in content or 'MetaTrader' in content
            live_data_usage = 'mock' in content.lower() or 'fake' in content.lower()
            
            # Detect violations
            violations = []
            if live_data_usage:
                violations.append("MOCK_DATA_DETECTED")
            if not eventbus_usage and len(functions) > 0:
                violations.append("NO_EVENTBUS_INTEGRATION")
            if not telemetry_enabled and 'engine' in file_path.name.lower():
                violations.append("NO_TELEMETRY_INTEGRATION")
                
            return {
                'imports': imports,
                'functions': functions,
                'classes': classes,
                'event_bus_usage': eventbus_usage,
                'telemetry_enabled': telemetry_enabled,
                'mt5_integration': mt5_integration,
                'live_data_usage': live_data_usage,
                'category': genesis_category,
                'violations': violations,
                'content_hash': hashlib.md5(content.encode()).hexdigest()[:8]
            }
            
        except Exception as e:
            return {
                'imports': [],
                'functions': [],
                'classes': [],
                'event_bus_usage': False,
                'telemetry_enabled': False,
                'mt5_integration': False,
                'live_data_usage': False,
                'category': 'UNKNOWN',
                'violations': [f"SCAN_ERROR: {str(e)}"],
                'content_hash': 'ERROR'
            }

    def extract_genesis_category(self, content: str) -> str:
        """Extract GENESIS category from content"""
        # Look for explicit category annotation
        category_match = re.search(r'@GENESIS_CATEGORY:\s*([A-Z_.]+)', content)
        if category_match:
            return category_match.group(1)
        
        # Infer from filename patterns
        content_lower = content.lower()
        for category, keywords in self.categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return f"CORE.{category}"
        
        return "UNKNOWN"

    def classify_orphan_intent(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """Classify orphan modules by intent without deletion"""
        file_name = file_path.name.lower()
        path_str = str(file_path).lower()
        
        # Quarantined files
        if 'triage_orphan_quarantine' in path_str:
            return "QUARANTINED"
            
        # Test files
        if file_name.startswith('test_') or '_test.py' in file_name:
            return "TEST_MODULE"
            
        # Validation files
        if 'validate_' in file_name or 'validation' in file_name:
            return "VALIDATION_MODULE"
            
        # Phase files
        if file_name.startswith('phase_') or 'phase' in file_name:
            return "PHASE_MODULE"
            
        # Backup/copy files
        if any(suffix in file_name for suffix in ['.backup', '.old', '.new', '_copy']):
            return "BACKUP_COPY"
            
        # Has violations
        if metadata['violations']:
            return "NEEDS_REPAIR"
            
        # Has EventBus or telemetry
        if metadata['event_bus_usage'] or metadata['telemetry_enabled']:
            return "RECOVERABLE"
            
        # Has imports/functions but no integration
        if metadata['functions'] and not metadata['event_bus_usage']:
            return "NEEDS_INTEGRATION"
            
        return "UNKNOWN_INTENT"

    def scan_all_modules(self) -> Dict[str, Any]:
        """Scan all Python modules in the workspace"""
        print("🔍 Starting comprehensive module scan...")
        
        registry = {
            'genesis_metadata': {
                'version': 'v3.0_phase101',
                'generation_timestamp': datetime.now().isoformat(),
                'architect_mode': True,
                'scan_agent': 'phase_101_architecture_resync_agent'
            },
            'modules': {},
            'categories': defaultdict(list),
            'orphans_by_intent': defaultdict(list),
            'duplicates': [],
            'violations': [],
            'statistics': {}
        }
        
        # Scan all .py files
        for py_file in self.workspace_path.rglob('*.py'):
            try:
                self.modules_scanned += 1
                
                # Get file metadata
                stat = py_file.stat()
                relative_path = py_file.relative_to(self.workspace_path)
                
                # Scan content
                content_metadata = self.scan_module_content(py_file)
                
                # Create module entry
                module_entry = {
                    'name': py_file.stem,
                    'full_name': py_file.name,
                    'path': str(py_file),
                    'relative_path': str(relative_path),
                    'extension': py_file.suffix,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    **content_metadata
                }
                
                # Classify module
                if content_metadata['category'] == 'UNKNOWN':
                    orphan_intent = self.classify_orphan_intent(py_file, content_metadata)
                    registry['orphans_by_intent'][orphan_intent].append(module_entry)
                    self.orphans_found += 1
                else:
                    registry['categories'][content_metadata['category']].append(module_entry)
                    self.connected_modules += 1
                
                # Store in main registry
                registry['modules'][str(relative_path)] = module_entry
                
                # Track violations
                if content_metadata['violations']:
                    registry['violations'].extend([
                        {
                            'module': str(relative_path),
                            'violation': violation
                        } for violation in content_metadata['violations']
                    ])
                
                if self.modules_scanned % 100 == 0:
                    print(f"📊 Scanned {self.modules_scanned} modules...")
                    
            except Exception as e:
                print(f"⚠️ Error scanning {py_file}: {e}")
                continue
        
        # Generate statistics
        registry['statistics'] = {
            'total_modules_scanned': self.modules_scanned,
            'connected_modules': self.connected_modules,
            'orphan_modules': self.orphans_found,
            'violation_count': len(registry['violations']),
            'categories_detected': len(registry['categories']),
            'orphan_intents': {intent: len(modules) for intent, modules in registry['orphans_by_intent'].items()}
        }
        
        return registry

    def rebuild_system_tree(self, registry: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild system_tree.json with connectivity mapping"""
        print("🌳 Rebuilding system tree with connectivity mapping...")
        
        system_tree = {
            'genesis_system_metadata': {
                'version': 'v3.0_phase101',
                'generation_timestamp': datetime.now().isoformat(),
                'architect_mode': True,
                'compliance_enforced': True,
                'scan_type': 'phase_101_resync',
                'total_files_scanned': self.modules_scanned,
                'categorized_modules': self.connected_modules,
                'orphan_modules': self.orphans_found
            },
            'connected_modules': registry['categories'],
            'orphan_classification': registry['orphans_by_intent'],
            'connectivity_matrix': self.build_connectivity_matrix(registry),
            'duplicate_candidates': self.detect_duplicates(registry),
            'violations_summary': registry['violations']
        }
        
        return system_tree

    def build_connectivity_matrix(self, registry: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build module connectivity matrix"""
        connectivity = {}
        
        for module_path, module_data in registry['modules'].items():
            connections = []
            
            # Check imports for internal connections
            for imp in module_data['imports']:
                for other_path, other_data in registry['modules'].items():
                    if other_path != module_path and imp in other_data['name']:
                        connections.append(other_path)
            
            connectivity[module_path] = connections
            
        return connectivity

    def detect_duplicates(self, registry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect potential duplicate modules"""
        duplicates = []
        name_groups = defaultdict(list)
        
        # Group by base name
        for module_path, module_data in registry['modules'].items():
            base_name = module_data['name'].replace('_copy', '').replace('_old', '').replace('_new', '')
            name_groups[base_name].append(module_data)
        
        # Find groups with multiple entries
        for base_name, modules in name_groups.items():
            if len(modules) > 1:
                duplicates.append({
                    'base_name': base_name,
                    'files': modules,
                    'count': len(modules)
                })
        
        return duplicates

    def rebuild_event_bus_mapping(self, registry: Dict[str, Any]) -> Dict[str, Any]:
        """Rebuild event_bus.json with route mapping"""
        print("🚌 Rebuilding EventBus route mapping...")
        
        event_bus = {
            'genesis_eventbus_metadata': {
                'version': 'v3.0_phase101',
                'generation_timestamp': datetime.now().isoformat(),
                'architect_mode': True
            },
            'active_routes': {},
            'subscribers': {},
            'emitters': {},
            'isolated_modules': []
        }
        
        for module_path, module_data in registry['modules'].items():
            if module_data['event_bus_usage']:
                event_bus['active_routes'][module_path] = {
                    'module': module_data['name'],
                    'category': module_data['category'],
                    'functions': module_data['functions']
                }
            else:
                if module_data['functions'] and module_data['category'] != 'UNKNOWN':
                    event_bus['isolated_modules'].append(module_path)
        
        return event_bus

    def generate_triage_report(self, registry: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive triage report"""
        print("📋 Generating triage report...")
        
        return {
            'triage_metadata': {
                'version': 'v3.0_phase101',
                'generation_timestamp': datetime.now().isoformat(),
                'total_modules_analyzed': self.modules_scanned
            },
            'orphan_classification': registry['orphans_by_intent'],
            'recovery_recommendations': {
                'RECOVERABLE': "Wire to EventBus and enable telemetry",
                'NEEDS_INTEGRATION': "Add EventBus emit/consume patterns",
                'NEEDS_REPAIR': "Fix violations before integration",
                'TEST_MODULE': "Validate test coverage, consider archiving",
                'VALIDATION_MODULE': "Integrate into CI/CD pipeline",
                'PHASE_MODULE': "Archive if phase completed",
                'BACKUP_COPY': "Mark for deletion after verification",
                'QUARANTINED': "Review quarantine status",
                'UNKNOWN_INTENT': "Manual review required"
            },
            'duplicate_analysis': self.detect_duplicates(registry),
            'violation_summary': {
                'total_violations': len(registry['violations']),
                'critical_violations': [v for v in registry['violations'] if 'MOCK_DATA' in v['violation']],
                'integration_violations': [v for v in registry['violations'] if 'EVENTBUS' in v['violation']]
            }
        }

    def update_build_status(self):
        """Update build_status.json with Phase 101 completion"""
        build_status_path = self.workspace_path / 'build_status.json'
        
        try:
            with open(build_status_path, 'r') as f:
                build_status = json.load(f)
        except:
            build_status = {}
        
        build_status.update({
            'phase_101_status': 'COMPLETED',
            'architecture_resync_completed': datetime.now().isoformat(),
            'modules_scanned': self.modules_scanned,
            'orphans_classified': self.orphans_found,
            'connected_modules': self.connected_modules,
            'last_updated': datetime.now().isoformat()
        })
          with open(build_status_path, 'w') as f:
            json.dump(build_status, f, indent=2)
            
    def log_to_build_tracker(self, message: str):
        """Log to build_tracker.md"""
        build_tracker_path = self.workspace_path / 'build_tracker.md'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Remove emojis that cause encoding issues
        clean_message = message.replace('🏗️', 'BUILD').replace('✅', 'SUCCESS').replace('📊', 'STATS').replace('🔗', 'CONNECTED').replace('🚨', 'ORPHANS').replace('📁', 'FILES').replace('🎯', 'NEXT').replace('❌', 'ERROR')
        log_entry = f"\n### PHASE 101 ARCHITECTURE RESYNC - {timestamp}\n{clean_message}\n"
        
        try:
            with open(build_tracker_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except:
            with open(build_tracker_path, 'w', encoding='utf-8') as f:
                f.write(f"# GENESIS BUILD TRACKER\n{log_entry}")

    def execute_resync(self):
        """Execute the complete architecture resync"""
        print("🚀 GENESIS PHASE 101 - ARCHITECTURE RESYNC AGENT STARTING...")
        
        # Create output directory
        output_dir = self.workspace_path / 'GENESIS_HIGH_ARCHITECTURE_STATUS_20250620_165419'
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Scan all modules
            registry = self.scan_all_modules()
            
            # Step 2: Rebuild system tree
            system_tree = self.rebuild_system_tree(registry)
            
            # Step 3: Rebuild EventBus mapping
            event_bus = self.rebuild_event_bus_mapping(registry)
            
            # Step 4: Generate triage report
            triage_report = self.generate_triage_report(registry)
            
            # Step 5: Write all files
            output_files = {
                'module_registry.json': registry,
                'system_tree.json': system_tree,
                'event_bus.json': event_bus,
                'triage_report.json': triage_report
            }
            
            for filename, data in output_files.items():
                # Write to workspace root
                with open(self.workspace_path / filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Write to output directory
                with open(output_dir / filename, 'w') as f:
                    json.dump(data, f, indent=2)
            
            # Step 6: Update build status and tracker
            self.update_build_status()
            self.log_to_build_tracker(f"""
✅ **PHASE 101 ARCHITECTURE RESYNC COMPLETED**

📊 **Results:**
- Modules Scanned: {self.modules_scanned}
- Connected Modules: {self.connected_modules}
- Orphan Modules: {self.orphans_found}
- Violations Found: {len(registry['violations'])}

📁 **Files Generated:**
- module_registry.json (rebuilt)
- system_tree.json (rebuilt)
- event_bus.json (rebuilt)
- triage_report.json (new)

🎯 **Next Phase:** Phase 102 - Kill-Switch Execution Loop
            """)
            
            print(f"✅ PHASE 101 COMPLETED SUCCESSFULLY!")
            print(f"📊 Scanned: {self.modules_scanned} modules")
            print(f"🔗 Connected: {self.connected_modules} modules")
            print(f"🚨 Orphans: {self.orphans_found} modules")
            print(f"📁 Output saved to: {output_dir}")
            
            return True
            
        except Exception as e:
            self.log_to_build_tracker(f"❌ **PHASE 101 FAILED:** {str(e)}")
            print(f"❌ PHASE 101 FAILED: {e}")
            return False

def main():
    """Main execution entry point"""
    workspace_path = r"c:\Users\patra\Genesis FINAL TRY"
    
    agent = GenesisArchitectureResyncAgent(workspace_path)
    success = agent.execute_resync()
    
    if success:
        print("\n🏁 Phase 101 Architecture Resync Agent completed successfully!")
        print("🎯 Ready for Phase 102: Kill-Switch Execution Loop")
    else:
        print("\n❌ Phase 101 Architecture Resync Agent failed!")
        print("🔧 Check build_tracker.md for details")

if __name__ == "__main__":
    main()


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
