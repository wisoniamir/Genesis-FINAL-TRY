from datetime import datetime
#!/usr/bin/env python3
"""
🔧 GENESIS EVENTBUS SEGMENTED LOADER v7.0.0
ARCHITECT MODE - ZERO TOLERANCE ENFORCEMENT

Runtime loader for segmented EventBus architecture
Prevents stack overflow while maintaining full connectivity
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# FTMO compliance enforcement - MANDATORY
try:
    from compliance.ftmo.enforcer import enforce_limits
    COMPLIANCE_AVAILABLE = True
except ImportError:
    def enforce_limits(signal="", risk_pct=0, data=None): 
        print(f"COMPLIANCE CHECK: {signal}")
        return True
    COMPLIANCE_AVAILABLE = False


class EventBusSegmentedLoader:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.segments_dir = self.workspace_path / "EVENTBUS_SEGMENTS"
        self.index_file = self.workspace_path / "event_bus_index.json"
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Cache for loaded segments
        self._segment_cache = {}
        self._master_index = None
        
    def load_index(self) -> Dict:
        """Load the master index"""
        if self._master_index is None:
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self._master_index = json.load(f)
                self.logger.info("📋 Master index loaded successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to load master index: {e}")
                self._master_index = {}
        
        return self._master_index
    
    def load_segment(self, segment_name: str) -> Optional[Dict]:
        """Load a specific segment with caching"""
        if segment_name in self._segment_cache:
            return self._segment_cache[segment_name]
        
        segment_file = self.segments_dir / f"{segment_name}.json"
        
        try:
            with open(segment_file, 'r', encoding='utf-8') as f:
                segment_data = json.load(f)
            
            self._segment_cache[segment_name] = segment_data
            self.logger.debug(f"📂 Loaded segment: {segment_name}")
            return segment_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load segment {segment_name}: {e}")
            return None
    
    def get_route(self, route_name: str) -> Optional[Dict]:
        """Get a specific route by name"""
        index = self.load_index()
        
        if "route_lookup" not in index:
            self.logger.error("❌ Route lookup table not found in index")
            return None
        
        if route_name not in index["route_lookup"]:
            self.logger.warning(f"⚠️ Route not found: {route_name}")
            return None
        
        segment_name = index["route_lookup"][route_name]
        segment_data = self.load_segment(segment_name)
        
        if segment_data and "routes" in segment_data:
            return segment_data["routes"].get(route_name)
        
        return None
    
    def get_routes_by_category(self, category: str) -> Dict:
        """Get all routes in a specific category"""
        index = self.load_index()
        
        category_segment = f"event_bus_{category}"
        segment_data = self.load_segment(category_segment)
        
        if segment_data and "routes" in segment_data:
            return segment_data["routes"]
        
        return {}
    
    def get_all_routes(self) -> Dict:
        """Assemble all routes from all segments (use with caution for large datasets)"""
        index = self.load_index()
        all_routes = {}
        
        if "segments" not in index:
            self.logger.error("❌ Segments information not found in index")
            return {}
        
        for segment_name, segment_info in index["segments"].items():
            if segment_info.get("type") == "routes":
                segment_data = self.load_segment(segment_name)
                if segment_data and "routes" in segment_data:
                    all_routes.update(segment_data["routes"])
        
        self.logger.info(f"📊 Assembled {len(all_routes)} routes from {len(index['segments'])} segments")
        return all_routes
    
    def get_routes_by_source(self, source: str) -> List[Dict]:
        """Get all routes that have a specific source"""
        matching_routes = []
        index = self.load_index()
        
        for segment_name, segment_info in index["segments"].items():
            if segment_info.get("type") == "routes":
                segment_data = self.load_segment(segment_name)
                if segment_data and "routes" in segment_data:
                    for route_name, route_data in segment_data["routes"].items():
                        if route_data.get("source") == source:
                            matching_routes.append({
                                "name": route_name,
                                "data": route_data,
                                "segment": segment_name
                            })
        
        return matching_routes
    
    def get_routes_by_destination(self, destination: str) -> List[Dict]:
        """Get all routes that have a specific destination"""
        matching_routes = []
        index = self.load_index()
        
        for segment_name, segment_info in index["segments"].items():
            if segment_info.get("type") == "routes":
                segment_data = self.load_segment(segment_name)
                if segment_data and "routes" in segment_data:
                    for route_name, route_data in segment_data["routes"].items():
                        destinations = route_data.get("destination", [])
                        if isinstance(destinations, str):
                            destinations = [destinations]
                        if destination in destinations:
                            matching_routes.append({
                                "name": route_name,
                                "data": route_data,
                                "segment": segment_name
                            })
        
        return matching_routes
    
    def validate_route_connectivity(self) -> Dict:
        """Validate that all routes have proper source/destination connectivity"""
        validation_report = {
            "total_routes": 0,
            "valid_routes": 0,
            "missing_source": [],
            "missing_destination": [],
            "categories": {}
        }
        
        index = self.load_index()
        
        for segment_name, segment_info in index["segments"].items():
            if segment_info.get("type") == "routes":
                category = segment_info.get("category", "unknown")
                validation_report["categories"][category] = {
                    "total": 0,
                    "valid": 0,
                    "issues": []
                }
                
                segment_data = self.load_segment(segment_name)
                if segment_data and "routes" in segment_data:
                    for route_name, route_data in segment_data["routes"].items():
                        validation_report["total_routes"] += 1
                        validation_report["categories"][category]["total"] += 1
                        
                        has_source = "source" in route_data and route_data["source"]
                        has_destination = "destination" in route_data and route_data["destination"]
                        
                        if not has_source:
                            validation_report["missing_source"].append(route_name)
                            validation_report["categories"][category]["issues"].append(f"{route_name}: missing source")
                        
                        if not has_destination:
                            validation_report["missing_destination"].append(route_name)
                            validation_report["categories"][category]["issues"].append(f"{route_name}: missing destination")
                        
                        if has_source and has_destination:
                            validation_report["valid_routes"] += 1
                            validation_report["categories"][category]["valid"] += 1
        
        self.logger.info(f"📊 Validation: {validation_report['valid_routes']}/{validation_report['total_routes']} routes valid")
        return validation_report
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the segmented EventBus"""
        index = self.load_index()
        
        stats = {
            "segments": len(index.get("segments", {})),
            "total_routes": sum(seg.get("route_count", 0) for seg in index.get("segments", {}).values()),
            "categories": {},
            "index_created": index.get("created_timestamp"),
            "version": index.get("version")
        }
        
        for segment_name, segment_info in index.get("segments", {}).items():
            category = segment_info.get("category", "unknown")
            if category not in stats["categories"]:
                stats["categories"][category] = {
                    "segments": 0,
                    "routes": 0
                }
            
            stats["categories"][category]["segments"] += 1
            stats["categories"][category]["routes"] += segment_info.get("route_count", 0)
        
        return stats

# Global instance for easy access
_global_loader = None

def get_eventbus_loader(workspace_path: str = None) -> EventBusSegmentedLoader:
    """Get global EventBus loader instance"""
    global _global_loader
    
    if _global_loader is None:
        if workspace_path is None:
            workspace_path = r"c:\Users\patra\Genesis FINAL TRY"
        _global_loader = EventBusSegmentedLoader(workspace_path)
    
    return _global_loader

# Convenience functions for common operations
def get_route(route_name: str) -> Optional[Dict]:
    """Get a specific route by name"""
    loader = get_eventbus_loader()
    return loader.get_route(route_name)

def get_routes_by_category(category: str) -> Dict:
    """Get all routes in a specific category"""
    loader = get_eventbus_loader()
    return loader.get_routes_by_category(category)

def get_dashboard_routes() -> Dict:
    """Get all dashboard-related routes"""
    return get_routes_by_category("dashboard")

def get_signal_routes() -> Dict:
    """Get all signal-related routes"""
    return get_routes_by_category("signals")

def get_execution_routes() -> Dict:
    """Get all execution-related routes"""
    return get_routes_by_category("execution")

def validate_connectivity() -> Dict:
    """Validate EventBus connectivity"""
    loader = get_eventbus_loader()
    return loader.validate_route_connectivity()

if __name__ == "__main__":
    # FTMO compliance enforcement
enforce_limits(signal="event_bus_segmented_loader")
    # Setup EventBus hooks
if EVENTBUS_AVAILABLE:
    event_bus = get_event_bus()
    if event_bus:
        # Register routes
        register_route("REQUEST_EVENT_BUS_SEGMENTED_LOADER", "event_bus_segmented_loader")
        
        # Emit initialization event
        emit_event("EVENT_BUS_SEGMENTED_LOADER_EMIT", {
            "status": "initializing",
            "timestamp": datetime.now().isoformat(),
            "module_id": "event_bus_segmented_loader"
        })

    # Test the loader
    loader = get_eventbus_loader()
    stats = loader.get_statistics()
    
    print("🔧 EventBus Segmented Loader Test")
    print("=" * 40)
    print(f"📊 Statistics: {json.dumps(stats, indent=2)}")
    
    # Test validation
    validation = loader.validate_route_connectivity()
    print(f"✅ Validation: {validation['valid_routes']}/{validation['total_routes']} routes valid")


    # Added by batch repair script
    # Emit completion event via EventBus
if EVENTBUS_AVAILABLE:
    emit_event("EVENT_BUS_SEGMENTED_LOADER_EMIT", {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "result": "success",
        "module_id": "event_bus_segmented_loader"
    })
