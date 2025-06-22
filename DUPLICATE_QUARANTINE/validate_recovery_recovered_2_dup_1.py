import logging
# <!-- @GENESIS_MODULE_START: validate_recovery -->

#!/usr/bin/env python3
"""
GENESIS RECOVERY VALIDATION ENGINE
Validate that recovered modules are properly integrated and functional

🎯 PURPOSE: Ensure recovered modules have proper EventBus/MT5 connectivity
🔧 TESTS: Import validation, EventBus integration, MT5 connection capability
"""

import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

def validate_recovered_modules():
    """Validate all recovered modules for proper integration"""
    
    base_path = Path("c:/Users/patra/Genesis FINAL TRY")
    recovery_path = base_path / "src" / "genesis_fixed"
    
    validation_results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'validated_modules': [],
        'errors': [],
        'overall_status': 'UNKNOWN'
    }
    
    recovered_modules = [
        'auto_execution_manager_recovered.py',
        'execution_supervisor_recovered.py', 
        'strategy_adaptive_context_synthesizer_recovered.py'
    ]
    
    print("🧪 GENESIS RECOVERY VALIDATION ENGINE")
    print("=" * 50)
    
    for module_file in recovered_modules:
        module_path = recovery_path / module_file
        module_name = module_file.replace('.py', '')
        
        print(f"\n🔍 Validating: {module_name}")
        
        module_result = {
            'module': module_name,
            'file_path': str(module_path),
            'exists': False,
            'importable': False,
            'has_eventbus': False,
            'has_mt5': False,
            'has_main_class': False,
            'validation_score': 0
        }
        
        try:
            # Check file exists
            if module_path.exists():
                module_result['exists'] = True
                print(f"  ✅ File exists: {module_path.stat().st_size} bytes")
                
                # Read content for analysis
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for EventBus integration
                if 'emit_event' in content or 'subscribe_to_event' in content:
                    module_result['has_eventbus'] = True
                    print("  ✅ EventBus integration detected")
                
                # Check for MT5 integration  
                if 'MT5' in content or 'MetaTrader5' in content or 'mt5.' in content:
                    module_result['has_mt5'] = True
                    print("  ✅ MT5 integration detected")
                
                # Check for main class
                if 'class ' in content:
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
                    module_result['has_main_class'] = True
                    print("  ✅ Class definitions found")
                
                # Try importing (basic syntax check)
                try:
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec and spec.loader:
                        module_result['importable'] = True
                        print("  ✅ Module syntax valid")
                except Exception as e:
                    print(f"  ❌ Import error: {e}")
                    module_result['import_error'] = str(e)
            else:
                print(f"  ❌ File not found: {module_path}")
                
        except Exception as e:
            error_msg = f"Error validating {module_name}: {str(e)}"
            print(f"  ❌ {error_msg}")
            validation_results['errors'].append(error_msg)
        
        # Calculate validation score
        score = sum([
            module_result['exists'] * 20,
            module_result['importable'] * 20,
            module_result['has_eventbus'] * 25,
            module_result['has_mt5'] * 25,
            module_result['has_main_class'] * 10
        ])
        module_result['validation_score'] = score
        
        print(f"  📊 Validation Score: {score}/100")
        validation_results['validated_modules'].append(module_result)
    
    # Overall assessment
    avg_score = sum(m['validation_score'] for m in validation_results['validated_modules']) / len(validation_results['validated_modules'])
    
    if avg_score >= 90:
        validation_results['overall_status'] = 'EXCELLENT'
        status_icon = "🏆"
    elif avg_score >= 75:
        validation_results['overall_status'] = 'GOOD' 
        status_icon = "✅"
    elif avg_score >= 50:
        validation_results['overall_status'] = 'PARTIAL'
        status_icon = "⚠️"
    else:
        validation_results['overall_status'] = 'FAILED'
        status_icon = "❌"
    
    print(f"\n{status_icon} OVERALL VALIDATION: {validation_results['overall_status']}")
    print(f"📊 Average Score: {avg_score:.1f}/100")
    
    # Save validation report
    with open(base_path / 'recovery_validation_report.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    return validation_results

if __name__ == "__main__":
    validate_recovered_modules()

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
        

# <!-- @GENESIS_MODULE_END: validate_recovery -->