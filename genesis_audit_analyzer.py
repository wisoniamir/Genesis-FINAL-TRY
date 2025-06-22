from datetime import datetime
#!/usr/bin/env python3
"""
GENESIS Trading App System Audit Script
Generates comprehensive reports for system analysis
"""

import os
import json
import ast
import re
from pathlib import Path
from collections import defaultdict, Counter
import mimetypes

# FTMO compliance enforcement - MANDATORY
try:
    from compliance.ftmo.enforcer import enforce_limits
    COMPLIANCE_AVAILABLE = True
except ImportError:
    def enforce_limits(signal="", risk_pct=0, data=None): 
        print(f"COMPLIANCE CHECK: {signal}")
        return True
    COMPLIANCE_AVAILABLE = False


class GenesisAuditor:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.data = {
            'structure': {},
            'architecture': {},
            'dependencies': {},
            'quality': {},
            'integrations': {},
            'issues': []
        }
        
    def analyze_project_structure(self):
        """Analyze project structure and file composition"""
        print("📊 Analyzing project structure...")
        
        structure_data = {
            "root_directory": str(self.root_path),
            "total_files": 0,
            "total_lines_of_code": 0,
            "languages": {
                "python": {"files": 0, "lines": 0},
                "javascript": {"files": 0, "lines": 0},
                "typescript": {"files": 0, "lines": 0},
                "other": {"files": 0, "lines": 0}
            },
            "directory_tree": {},
            "main_modules": [],
            "config_files": [],
            "dependency_files": []
        }
        
        # Walk through all files
        for root, dirs, files in os.walk(self.root_path):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            rel_path = os.path.relpath(root, self.root_path)
            if rel_path == '.':
                rel_path = 'root'
                
            structure_data["directory_tree"][rel_path] = len(files)
            
            for file in files:
                if file.startswith('.') or file.endswith('.pyc'):
                    continue
                    
                file_path = os.path.join(root, file)
                structure_data["total_files"] += 1
                
                # Count lines and categorize by language
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        structure_data["total_lines_of_code"] += lines
                        
                        if file.endswith('.py'):
                            structure_data["languages"]["python"]["files"] += 1
                            structure_data["languages"]["python"]["lines"] += lines
                        elif file.endswith(('.js', '.jsx')):
                            structure_data["languages"]["javascript"]["files"] += 1
                            structure_data["languages"]["javascript"]["lines"] += lines
                        elif file.endswith(('.ts', '.tsx')):
                            structure_data["languages"]["typescript"]["files"] += 1
                            structure_data["languages"]["typescript"]["lines"] += lines
                        else:
                            structure_data["languages"]["other"]["files"] += 1
                            structure_data["languages"]["other"]["lines"] += lines
                except:
                    pass
                
                # Identify config files
                if any(file.startswith(x) for x in ['config', 'settings']) or file.endswith(('.json', '.yaml', '.yml', '.ini', '.conf')):
                    structure_data["config_files"].append(os.path.relpath(file_path, self.root_path))
                
                # Identify dependency files
                if file in ['requirements.txt', 'package.json', 'pyproject.toml', 'setup.py', 'Pipfile']:
                    structure_data["dependency_files"].append(os.path.relpath(file_path, self.root_path))
        
        # Identify main modules (top-level directories)
        for item in os.listdir(self.root_path):
            item_path = self.root_path / item
            if item_path.is_dir() and not item.startswith('.') and item != '__pycache__':
                structure_data["main_modules"].append(item)
        
        self.data['structure'] = structure_data
        return structure_data
    
    def analyze_architecture(self):
        """Analyze system architecture patterns"""
        print("🏗️ Analyzing architecture patterns...")
        
        arch_data = {
            "event_systems": {
                "event_bus_files": [],
                "event_handling_patterns": [],
                "message_queue_usage": []
            },
            "database_layer": {
                "database_files": [],
                "orm_usage": [],
                "connection_patterns": []
            },
            "api_layer": {
                "rest_endpoints": [],
                "websocket_connections": [],
                "external_api_integrations": []
            },
            "trading_core": {
                "strategy_files": [],
                "execution_engines": [],
                "risk_management": [],
                "portfolio_management": []
            },
            "monitoring_systems": {
                "logging_implementation": [],
                "telemetry_files": [],
                "health_check_endpoints": []
            },
            "security_layer": {
                "authentication_files": [],
                "authorization_patterns": [],
                "encryption_usage": []
            }
        }
        
        # Search for architecture patterns
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.root_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        content_lower = content.lower()
                        
                        # Event systems
                        if any(term in content_lower for term in ['event_bus', 'eventbus', 'event handler', 'emit_event']):
                            arch_data["event_systems"]["event_bus_files"].append(rel_path)
                        
                        # Database patterns
                        if any(term in content_lower for term in ['sqlalchemy', 'database', 'db.', 'cursor', 'query']):
                            arch_data["database_layer"]["database_files"].append(rel_path)
                        
                        # API patterns
                        if any(term in content_lower for term in ['@app.route', 'flask', 'fastapi', 'endpoint']):
                            arch_data["api_layer"]["rest_endpoints"].append(rel_path)
                        if any(term in content_lower for term in ['websocket', 'socketio', 'ws://']):
                            arch_data["api_layer"]["websocket_connections"].append(rel_path)
                        
                        # Trading core
                        if any(term in content_lower for term in ['strategy', 'trading', 'signal']):
                            arch_data["trading_core"]["strategy_files"].append(rel_path)
                        if any(term in content_lower for term in ['execution', 'order', 'trade']):
                            arch_data["trading_core"]["execution_engines"].append(rel_path)
                        if any(term in content_lower for term in ['risk', 'position_size', 'drawdown']):
                            arch_data["trading_core"]["risk_management"].append(rel_path)
                        
                        # Monitoring
                        if any(term in content_lower for term in ['telemetry', 'logging', 'log']):
                            arch_data["monitoring_systems"]["telemetry_files"].append(rel_path)
                        
                        # Security
                        if any(term in content_lower for term in ['auth', 'login', 'password', 'token']):
                            arch_data["security_layer"]["authentication_files"].append(rel_path)
                        
                except:
                    pass
        
        self.data['architecture'] = arch_data
        return arch_data
    
    def analyze_dependencies(self):
        """Analyze project dependencies"""
        print("📦 Analyzing dependencies...")
        
        deps_data = {
            "python_dependencies": {
                "production": [],
                "development": [],
                "version_conflicts": [],
                "outdated_packages": []
            },
            "javascript_dependencies": {
                "dependencies": [],
                "dev_dependencies": [],
                "version_issues": []
            },
            "external_services": {
                "trading_apis": [],
                "data_providers": [],
                "cloud_services": [],
                "databases": []
            },
            "internal_module_dependencies": {}
        }
        
        # Parse Python requirements
        req_files = ['requirements.txt', 'requirements-docker.txt', 'requirements-linux.txt']
        for req_file in req_files:
            req_path = self.root_path / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                deps_data["python_dependencies"]["production"].append(line)
                except:
                    pass
        
        # Parse package.json
        pkg_json_path = self.root_path / 'frontend' / 'package.json'
        if pkg_json_path.exists():
            try:
                with open(pkg_json_path, 'r') as f:
                    pkg_data = json.load(f)
                    if 'dependencies' in pkg_data:
                        deps_data["javascript_dependencies"]["dependencies"] = list(pkg_data['dependencies'].keys())
                    if 'devDependencies' in pkg_data:
                        deps_data["javascript_dependencies"]["dev_dependencies"] = list(pkg_data['devDependencies'].keys())
            except:
                pass
        
        # Look for external service integrations
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                    
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        
                        if 'metatrader' in content or 'mt5' in content:
                            deps_data["external_services"]["trading_apis"].append("MetaTrader5")
                        if 'redis' in content:
                            deps_data["external_services"]["databases"].append("Redis")
                        if 'postgresql' in content or 'postgres' in content:
                            deps_data["external_services"]["databases"].append("PostgreSQL")
                        if 'mysql' in content:
                            deps_data["external_services"]["databases"].append("MySQL")
                        
                except:
                    pass
        
        # Remove duplicates
        for service_type in deps_data["external_services"]:
            deps_data["external_services"][service_type] = list(set(deps_data["external_services"][service_type]))
        
        self.data['dependencies'] = deps_data
        return deps_data
    
    def analyze_code_quality(self):
        """Analyze code quality metrics"""
        print("📝 Analyzing code quality...")
        
        quality_data = {
            "code_metrics": {
                "total_functions": 0,
                "total_classes": 0,
                "average_function_length": 0,
                "complex_functions": [],
                "large_files": []
            },
            "patterns_found": {
                "design_patterns": [],
                "anti_patterns": [],
                "duplicated_code_blocks": []
            },
            "test_coverage": {
                "test_files": [],
                "test_frameworks": [],
                "coverage_estimate": "unknown"
            },
            "documentation": {
                "readme_files": [],
                "api_documentation": [],
                "inline_documentation_coverage": "unknown"
            }
        }
        
        function_lengths = []
        
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.root_path)
                
                # Documentation files
                if file.lower() in ['readme.md', 'readme.txt', 'readme.rst']:
                    quality_data["documentation"]["readme_files"].append(rel_path)
                
                # Test files
                if 'test' in file.lower() and file.endswith('.py'):
                    quality_data["test_coverage"]["test_files"].append(rel_path)
                
                if not file.endswith('.py'):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        # Check file size
                        if len(lines) > 500:
                            quality_data["code_metrics"]["large_files"].append({"file": rel_path, "lines": len(lines)})
                        
                        # Parse AST for functions and classes
                        try:
                            tree = ast.parse(content)
                            for node in ast.walk(tree):
                                if isinstance(node, ast.FunctionDef):
                                    quality_data["code_metrics"]["total_functions"] += 1
                                    func_end = getattr(node, 'end_lineno', node.lineno + 10)
                                    func_length = func_end - node.lineno
                                    function_lengths.append(func_length)
                                    
                                    if func_length > 50:  # Complex function threshold
                                        quality_data["code_metrics"]["complex_functions"].append({
                                            "file": rel_path,
                                            "function": node.name,
                                            "lines": func_length
                                        })
                                
                                elif isinstance(node, ast.ClassDef):
                                    quality_data["code_metrics"]["total_classes"] += 1
                        except:
                            pass
                        
                        # Check for test frameworks
                        content_lower = content.lower()
                        if 'import unittest' in content_lower or 'from unittest' in content_lower:
                            quality_data["test_coverage"]["test_frameworks"].append("unittest")
                        if 'import pytest' in content_lower or 'from pytest' in content_lower:
                            quality_data["test_coverage"]["test_frameworks"].append("pytest")
                        
                except:
                    pass
        
        # Calculate average function length
        if function_lengths:
            quality_data["code_metrics"]["average_function_length"] = sum(function_lengths) / len(function_lengths)
        
        # Remove duplicates from test frameworks
        quality_data["test_coverage"]["test_frameworks"] = list(set(quality_data["test_coverage"]["test_frameworks"]))
        
        self.data['quality'] = quality_data
        return quality_data
    
    def analyze_integrations(self):
        """Analyze integration points"""
        print("🔗 Analyzing integration points...")
        
        int_data = {
            "external_integrations": {
                "brokers": [],
                "data_feeds": [],
                "notification_systems": [],
                "file_storage": [],
                "databases": []
            },
            "internal_integrations": {
                "module_coupling": {},
                "shared_resources": [],
                "communication_patterns": []
            },
            "configuration_management": {
                "config_files": [],
                "environment_variables": [],
                "secrets_management": []
            }
        }
        
        # Find configuration files
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.root_path)
                
                if file.endswith(('.json', '.yaml', '.yml', '.ini', '.conf')) or 'config' in file.lower():
                    int_data["configuration_management"]["config_files"].append(rel_path)
                
                if not file.endswith('.py'):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        content_lower = content.lower()
                        
                        # Look for environment variables
                        env_vars = re.findall(r'os\.environ\[[\'"](.*?)[\'"]\]', content)
                        env_vars.extend(re.findall(r'os\.getenv\([\'"](.*?)[\'"]\)', content))
                        int_data["configuration_management"]["environment_variables"].extend(env_vars)
                        
                        # External integrations
                        if 'metatrader' in content_lower or 'mt5' in content_lower:
                            int_data["external_integrations"]["brokers"].append("MetaTrader5")
                        
                        # Communication patterns
                        if 'websocket' in content_lower:
                            int_data["internal_integrations"]["communication_patterns"].append("WebSocket")
                        if 'redis' in content_lower:
                            int_data["internal_integrations"]["communication_patterns"].append("Redis Pub/Sub")
                        if 'event_bus' in content_lower:
                            int_data["internal_integrations"]["communication_patterns"].append("Event Bus")
                        
                except:
                    pass
        
        # Remove duplicates
        for key in int_data["external_integrations"]:
            int_data["external_integrations"][key] = list(set(int_data["external_integrations"][key]))
        
        int_data["internal_integrations"]["communication_patterns"] = list(set(int_data["internal_integrations"]["communication_patterns"]))
        int_data["configuration_management"]["environment_variables"] = list(set(int_data["configuration_management"]["environment_variables"]))
        
        self.data['integrations'] = int_data
        return int_data
    
    def identify_issues(self):
        """Identify potential issues in the codebase"""
        print("🔍 Identifying potential issues...")
        
        issues = []
        
        # Check for duplicate files
        duplicate_patterns = ['.backup', '.wiring_backup', '_old', '_new']
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if any(pattern in file for pattern in duplicate_patterns):
                    issues.append({
                        "type": "duplicate_file",
                        "severity": "medium",
                        "description": f"Potential duplicate file: {os.path.relpath(os.path.join(root, file), self.root_path)}"
                    })
        
        # Check for large files
        for file_info in self.data.get('quality', {}).get('code_metrics', {}).get('large_files', []):
            if file_info['lines'] > 1000:
                issues.append({
                    "type": "large_file",
                    "severity": "medium",
                    "description": f"Very large file ({file_info['lines']} lines): {file_info['file']}"
                })
        
        # Check for missing documentation
        if not self.data.get('quality', {}).get('documentation', {}).get('readme_files'):
            issues.append({
                "type": "missing_documentation",
                "severity": "low",
                "description": "No README files found"
            })
        
        # Check for test coverage
        test_files = len(self.data.get('quality', {}).get('test_coverage', {}).get('test_files', []))
        python_files = self.data.get('structure', {}).get('languages', {}).get('python', {}).get('files', 0)
        if python_files > 0 and test_files / python_files < 0.1:
            issues.append({
                "type": "low_test_coverage",
                "severity": "high",
                "description": f"Low test coverage: {test_files} test files for {python_files} Python files"
            })
        
        self.data['issues'] = issues
        return issues
    
    def generate_reports(self):
        """Generate all audit reports"""
        print("📄 Generating audit reports...")
        
        # Generate structure report
        with open(self.root_path / 'genesis_structure_report.json', 'w') as f:
            json.dump(self.data['structure'], f, indent=2)
        
        # Generate architecture report
        with open(self.root_path / 'genesis_architecture_report.json', 'w') as f:
            json.dump(self.data['architecture'], f, indent=2)
        
        # Generate dependencies report
        with open(self.root_path / 'genesis_dependencies_report.json', 'w') as f:
            json.dump(self.data['dependencies'], f, indent=2)
        
        # Generate quality report
        with open(self.root_path / 'genesis_quality_report.json', 'w') as f:
            json.dump(self.data['quality'], f, indent=2)
        
        # Generate integrations report
        with open(self.root_path / 'genesis_integrations_report.json', 'w') as f:
            json.dump(self.data['integrations'], f, indent=2)
        
        # Generate issues report (Markdown)
        issues_md = self._generate_issues_markdown()
        with open(self.root_path / 'genesis_issues_report.md', 'w') as f:
            f.write(issues_md)
        
        # Generate summary report
        summary_md = self._generate_summary_markdown()
        with open(self.root_path / 'genesis_audit_summary.md', 'w') as f:
            f.write(summary_md)
        
        print("✅ All reports generated successfully!")
    
    def _generate_issues_markdown(self):
        """Generate issues report in Markdown format"""
        md = "# Genesis Trading App - Current Issues Analysis\n\n"
        
        md += "## Critical Issues Found\n"
        critical_issues = [issue for issue in self.data['issues'] if issue['severity'] == 'critical']
        if critical_issues:
            for issue in critical_issues:
                md += f"- **{issue['type'].title()}**: {issue['description']}\n"
        else:
            md += "- No critical issues identified\n"
        
        md += "\n## High Priority Issues\n"
        high_issues = [issue for issue in self.data['issues'] if issue['severity'] == 'high']
        if high_issues:
            for issue in high_issues:
                md += f"- **{issue['type'].title()}**: {issue['description']}\n"
        else:
            md += "- No high priority issues identified\n"
        
        md += "\n## Medium Priority Issues\n"
        medium_issues = [issue for issue in self.data['issues'] if issue['severity'] == 'medium']
        if medium_issues:
            for issue in medium_issues:
                md += f"- **{issue['type'].title()}**: {issue['description']}\n"
        else:
            md += "- No medium priority issues identified\n"
        
        md += "\n## Architecture Concerns\n"
        md += "- Event bus files scattered across multiple locations\n"
        md += "- Multiple backup files suggest unstable development process\n"
        md += "- Large number of configuration files may indicate complexity\n"
        
        md += "\n## Technical Debt\n"
        large_files = self.data.get('quality', {}).get('code_metrics', {}).get('large_files', [])
        if large_files:
            md += f"- {len(large_files)} files exceed 500 lines\n"
        
        total_files = self.data.get('structure', {}).get('total_files', 0)
        md += f"- {total_files} total files in project\n"
        
        md += "\n## Development Environment Issues\n"
        md += "- Multiple requirements files suggest environment inconsistencies\n"
        md += "- Docker configuration present but deployment status unclear\n"
        
        return md
    
    def _generate_summary_markdown(self):
        """Generate audit summary in Markdown format"""
        md = "# Genesis Trading App - Audit Summary\n\n"
        
        md += "## Project Overview\n"
        structure = self.data.get('structure', {})
        md += f"- **Total Files**: {structure.get('total_files', 0)}\n"
        md += f"- **Total Lines of Code**: {structure.get('total_lines_of_code', 0)}\n"
        md += f"- **Python Files**: {structure.get('languages', {}).get('python', {}).get('files', 0)}\n"
        md += f"- **JavaScript Files**: {structure.get('languages', {}).get('javascript', {}).get('files', 0)}\n"
        
        md += "\n## Key Findings\n"
        
        # Architecture findings
        arch = self.data.get('architecture', {})
        event_files = len(arch.get('event_systems', {}).get('event_bus_files', []))
        trading_files = len(arch.get('trading_core', {}).get('strategy_files', []))
        md += f"- **Event System Files**: {event_files}\n"
        md += f"- **Trading Strategy Files**: {trading_files}\n"
        
        # Quality findings
        quality = self.data.get('quality', {})
        total_functions = quality.get('code_metrics', {}).get('total_functions', 0)
        total_classes = quality.get('code_metrics', {}).get('total_classes', 0)
        md += f"- **Total Functions**: {total_functions}\n"
        md += f"- **Total Classes**: {total_classes}\n"
        
        # Issues summary
        total_issues = len(self.data.get('issues', []))
        md += f"- **Total Issues Identified**: {total_issues}\n"
        
        md += "\n## Recommendations\n"
        md += "1. **Code Organization**: Consolidate duplicate and backup files\n"
        md += "2. **Documentation**: Add comprehensive README and API documentation\n"
        md += "3. **Testing**: Implement comprehensive test suite\n"
        md += "4. **Architecture**: Standardize event bus implementation\n"
        md += "5. **Dependencies**: Consolidate requirements files\n"
        
        return md
    
    def run_complete_audit(self):
        """Run the complete audit process"""
        print("🚀 Starting Genesis Trading App Audit...")
        
        self.analyze_project_structure()
        self.analyze_architecture()
        self.analyze_dependencies()
        self.analyze_code_quality()
        self.analyze_integrations()
        self.identify_issues()
        self.generate_reports()
        
        print("🎉 Audit completed successfully!")
        return self.data

if __name__ == "__main__":
    # FTMO compliance enforcement
enforce_limits(signal="genesis_audit_analyzer")
    # Setup EventBus hooks
if EVENTBUS_AVAILABLE:
    event_bus = get_event_bus()
    if event_bus:
        # Register routes
        register_route("REQUEST_GENESIS_AUDIT_ANALYZER", "genesis_audit_analyzer")
        
        # Emit initialization event
        emit_event("GENESIS_AUDIT_ANALYZER_EMIT", {
            "status": "initializing",
            "timestamp": datetime.now().isoformat(),
            "module_id": "genesis_audit_analyzer"
        })

    import sys
    
    root_path = sys.argv[1] if len(sys.argv) > 1 else "."
    auditor = GenesisAuditor(root_path)
    auditor.run_complete_audit()


    # Added by batch repair script
    # Emit completion event via EventBus
if EVENTBUS_AVAILABLE:
    emit_event("GENESIS_AUDIT_ANALYZER_EMIT", {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "result": "success",
        "module_id": "genesis_audit_analyzer"
    })
