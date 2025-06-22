#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🛰️ GENESIS SYNC BEACON TEST v1.0.0
═══════════════════════════════════════════════════════════════════════════════════════

🧠 PURPOSE: Test deployment of sync beacon with existing GENESIS infrastructure
📡 ARCHITECT MODE v7.0.0 COMPLIANT | 🔗 EventBus Integrated | 📊 Telemetry Enabled

This is a simplified test version to verify the integration works correctly.
"""

import os
import sys
import json
from datetime import datetime, timezone

def main():
    """Test GENESIS Sync Beacon integration"""
    print("🛰️ GENESIS SYNC BEACON TEST v1.0.0")
    print("=" * 60)
    
    # Test 1: Check if we can import GENESIS core modules
    print("\n📦 Testing GENESIS Core Module Imports...")
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Test EventBus import
        try:
            from core.event_bus import EventBus
            print("✅ EventBus imported successfully")
            
            # Test EventBus instantiation
            event_bus = EventBus()
            print(f"✅ EventBus instantiated: {event_bus.module_id}")
            
        except Exception as e:
            print(f"❌ EventBus import/instantiation failed: {e}")
        
        # Test Telemetry import
        try:
            from core.telemetry import emit_telemetry, TelemetryManager
            print("✅ Telemetry imported successfully")
            
            # Test telemetry call
            emit_telemetry("test_sync_beacon", "test_event", {"status": "testing"})
            print("✅ Telemetry call successful")
            
        except Exception as e:
            print(f"❌ Telemetry import/call failed: {e}")
            
    except Exception as e:
        print(f"❌ Core module import failed: {e}")
    
    # Test 2: Check Google Drive dependencies
    print("\n📦 Testing Google Drive Dependencies...")
    
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        print("✅ Google Drive API libraries available")
    except ImportError as e:
        print(f"❌ Google Drive dependencies missing: {e}")
        print("📝 Run: pip install google-api-python-client google-auth")
    
    # Test 3: Check credentials setup
    print("\n🔐 Testing Credentials Setup...")
    
    credentials_path = "credentials/google_drive_service_account.json"
    template_path = "credentials/google_drive_service_account.json.template"
    
    if os.path.exists(credentials_path):
        print("✅ Service account credentials found")
        
        try:
            with open(credentials_path, 'r') as f:
                creds_data = json.load(f)
            
            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if field not in creds_data]
            
            if not missing_fields:
                print("✅ Credentials file appears valid")
            else:
                print(f"⚠️  Credentials missing fields: {missing_fields}")
                
        except Exception as e:
            print(f"⚠️  Credentials file format error: {e}")
            
    elif os.path.exists(template_path):
        print("⚠️  Only template file found. Replace with actual credentials.")
    else:
        print("❌ No credentials file found")
    
    # Test 4: Check system registration
    print("\n📋 Testing System Registration...")
    
    try:
        # Check module registry
        registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "module_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            if "genesis_sync_beacon" in registry.get("modules", {}):
                print("✅ Sync beacon registered in module registry")
            else:
                print("⚠️  Sync beacon not found in module registry")
        else:
            print("❌ Module registry not found")
            
        # Check system tree
        tree_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "system_tree.json")
        if os.path.exists(tree_path):
            with open(tree_path, 'r') as f:
                tree = json.load(f)
            
            # Look for sync beacon in any category
            found_in_tree = False
            for category, modules in tree.get("connected_modules", {}).items():
                if isinstance(modules, list):
                    for module in modules:
                        if module.get("name") == "genesis_sync_beacon":
                            found_in_tree = True
                            print(f"✅ Sync beacon found in system tree under {category}")
                            break
            
            if not found_in_tree:
                print("⚠️  Sync beacon not found in system tree")
        else:
            print("❌ System tree not found")
            
    except Exception as e:
        print(f"❌ System registration check failed: {e}")
    
    # Test 5: EventBus route testing
    print("\n🔗 Testing EventBus Integration...")
    
    try:
        from core.event_bus import EventBus
        event_bus = EventBus()
        
        # Test event emission
        test_events = [
            "drive_sync_beacon_test",
            "sync_test_completed"
        ]
        
        for event in test_events:
            event_bus.emit(event, {
                "test_id": "sync_beacon_integration_test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "testing"
            })
            print(f"✅ Event emitted: {event}")
        
        print(f"✅ EventBus integration test completed")
        
    except Exception as e:
        print(f"❌ EventBus integration test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print("✅ Tests completed")
    print("📋 Next steps:")
    print("   1. Install Google Drive dependencies if missing")
    print("   2. Replace credentials template with actual service account JSON")
    print("   3. Run full sync beacon: python genesis_sync_beacon.py")
    print("   4. Monitor GENESIS dashboard for sync beacon status")


if __name__ == "__main__":
    main()
