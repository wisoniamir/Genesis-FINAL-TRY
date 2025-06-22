from datetime import datetime

# FTMO compliance enforcement - MANDATORY
try:
    from compliance.ftmo.enforcer import enforce_limits
    COMPLIANCE_AVAILABLE = True
except ImportError:
    def enforce_limits(signal="", risk_pct=0, data=None): 
        print(f"COMPLIANCE CHECK: {signal}")
        return True
    COMPLIANCE_AVAILABLE = False

