# 🛡️ GENESIS GIT PRE-COMMIT HOOK ENFORCEMENT SYSTEM

## 📋 OVERVIEW

The GENESIS Git Pre-Commit Hook Enforcement System provides **zero-tolerance** protection for your Git repository by automatically running comprehensive system audits before every commit. This ensures that **ARCHITECT MODE v7.0.0 compliance** is maintained at all times.

## 🎯 PURPOSE

- **Prevent non-compliant code** from entering the repository
- **Enforce real data only** usage (no mocks, stubs, or simulated data)
- **Validate system integrity** before each commit
- **Maintain ARCHITECT MODE** compliance automatically
- **Block commits** when violations are detected

## 🔧 COMPONENTS

### 1. `audit_engine.py` - Comprehensive System Validator
**Location:** Project root  
**Purpose:** Performs deep system validation across all GENESIS components

**Validation Categories:**
- 📁 **Core Files Validation** - Ensures all critical system files exist and are valid
- 👻 **Orphan Module Detection** - Identifies disconnected or unregistered modules
- 🔗 **EventBus Connectivity** - Verifies all modules are properly wired
- 🚫 **Mock Data Scanning** - Detects any simulated or test-only data usage
- 📊 **Telemetry Validation** - Ensures monitoring systems are active
- 🌳 **System Tree Integrity** - Validates system architecture compliance
- 🔐 **ARCHITECT MODE Compliance** - Verifies zero-tolerance enforcement
- 📡 **Real Data Enforcement** - Ensures only live MT5 data is used

### 2. `.git/hooks/pre-commit` - Git Hook (Unix/Linux)
**Location:** `.git/hooks/pre-commit`  
**Purpose:** Bash-based pre-commit hook for Unix-like systems

### 3. `.git/hooks/pre-commit.bat` - Git Hook (Windows)
**Location:** `.git/hooks/pre-commit.bat`  
**Purpose:** Batch-based pre-commit hook for Windows systems

## 🚦 ENFORCEMENT RULES

### ✅ COMMIT APPROVED CONDITIONS
- All core files present and valid
- Zero orphan modules detected
- All modules EventBus-connected
- No mock data violations found
- Telemetry system active
- System tree integrity verified
- ARCHITECT MODE compliance confirmed
- Real data only validation passed

### 🚫 COMMIT BLOCKED CONDITIONS
- **Critical Violations:**
  - Missing or corrupted core files
  - Mock/simulated data detected
  - `real_data_passed = false` found
  - `ARCHITECT_LOCK_BROKEN` detected
  - Telemetry system disabled

- **High Priority Violations:**
  - Orphan modules present
  - EventBus routes with empty functions
  - System tree compliance violations

## 📊 AUDIT PROCESS

### 1. **Pre-Commit Trigger**
When you run `git commit`, the hook automatically executes:

```bash
git commit -m "Your commit message"
```

### 2. **Audit Execution**
The system runs comprehensive validation:

```
🛡️ GENESIS SYSTEM AUDIT — Pre-Commit Hook Activated
🔐 ARCHITECT MODE v7.0.0 — Zero Tolerance Enforcement
============================================================
🔍 Running comprehensive GENESIS system audit...
```

### 3. **Results Processing**
- **PASS:** Commit proceeds normally
- **FAIL:** Commit is blocked with detailed report

### 4. **Report Generation**
Detailed audit report saved to `audit_snapshot_report.md`

## 📝 AUDIT REPORT FORMAT

```markdown
# GENESIS SYSTEM AUDIT REPORT

## 🛡️ AUDIT SUMMARY - [timestamp]
**AUDIT STATUS:** PASSED/FAILED
**COMPLIANCE SCORE:** [0-100]%
**VIOLATIONS FOUND:** [count]
**ARCHITECT LOCK STATUS:** SECURE/COMPROMISED
**REAL DATA VALIDATION:** PASSED/FAILED

## 📊 VIOLATION DETAILS
[Detailed breakdown of any violations found]

## ✅/🚫 ENFORCEMENT DECISION
[Commit approval/rejection with remediation steps]
```

## 🔧 MANUAL OPERATIONS

### Run Audit Manually
```bash
python audit_engine.py
```

### Emergency Repair (if needed)
```bash
python emergency_architect_compliance_fixer.py
```

### View Last Audit Report
```bash
# Windows
type audit_snapshot_report.md

# Unix/Linux
cat audit_snapshot_report.md
```

## 🚨 VIOLATION REMEDIATION

When a commit is blocked, follow these steps:

### 1. **Review Audit Report**
```bash
# Check the generated report
cat audit_snapshot_report.md
```

### 2. **Fix Critical Violations**
- **Mock Data:** Remove all simulated/test data
- **Orphan Modules:** Connect to EventBus and system tree
- **Missing Files:** Restore required core files
- **ARCHITECT_LOCK_BROKEN:** Run emergency repair

### 3. **Run Emergency Repair (if needed)**
```bash
python emergency_architect_compliance_fixer.py
```

### 4. **Verify Fixes**
```bash
python audit_engine.py
```

### 5. **Retry Commit**
```bash
git commit -m "Your commit message"
```

## 🔍 TROUBLESHOOTING

### Hook Not Running
```bash
# Ensure hooks are executable (Unix/Linux)
chmod +x .git/hooks/pre-commit

# Verify Git hooks path
git config core.hooksPath .git/hooks
```

### Python Not Found
```bash
# Verify Python installation
python --version
# or
python3 --version
```

### Audit Engine Missing
```bash
# Verify audit engine exists
ls -la audit_engine.py

# If missing, restore from repository
```

## 🎯 BEST PRACTICES

### 1. **Run Audits Frequently**
```bash
# Before major changes
python audit_engine.py

# After resolving violations
python audit_engine.py
```

### 2. **Monitor Compliance Score**
- Aim for **100% compliance** before commits
- Address violations immediately
- Keep audit reports for documentation

### 3. **Emergency Procedures**
```bash
# If system becomes unusable
python emergency_architect_compliance_fixer.py

# Force rebuild system tree
python system_tree_rebuild_engine.py
```

## 📈 INTEGRATION WITH ARCHITECT MODE

The Git pre-commit hook system integrates seamlessly with:

- **🐺 GENESIS Watchdog System** - Continuous monitoring
- **🔐 ARCHITECT MODE v7.0.0** - Zero tolerance enforcement
- **📊 Telemetry System** - Real-time metrics
- **🔗 EventBus Network** - Module connectivity
- **🌳 System Tree** - Architecture validation

## 🔒 SECURITY FEATURES

- **Zero Tolerance Policy** - No exceptions for violations
- **Real Data Only** - Prevents test data in production
- **Audit Trail** - Complete logging of all checks
- **Automatic Blocking** - Cannot bypass without fixing violations
- **Comprehensive Validation** - Multi-layer security checks

## 📋 MAINTENANCE

### Regular Tasks
1. **Weekly:** Review audit reports for patterns
2. **Monthly:** Update violation patterns if needed
3. **Quarterly:** Verify hook integrity and performance

### Updates
- Hook files are version-controlled
- Audit engine updates automatically deploy
- Configuration changes require administrator approval

---

## 🎉 CONCLUSION

The GENESIS Git Pre-Commit Hook Enforcement System provides **institutional-grade protection** for your codebase by:

- **Preventing violations** before they enter the repository
- **Maintaining ARCHITECT MODE compliance** automatically
- **Enforcing real data only** usage consistently
- **Providing clear remediation** guidance when issues arise
- **Integrating seamlessly** with existing GENESIS systems

**Your repository is now protected by zero-tolerance enforcement!** 🛡️
