# GENESIS MODULE REPAIR & ACTIVATION REPORT
## Generated: 2025-06-22 18:21:15

## 📊 Summary

- **Quarantined Modules**: 6699
- **Repaired Modules**: 6699
- **Activated Modules**: 6699
- **Failed Modules**: 0

## 🔄 EventBus Routes

- **Total Routes**: 29
- **Trading Routes**:

### SIGNAL_BUY
- **Emitters**: strategy_engine
- **Listeners**: execution_engine

### SIGNAL_SELL
- **Emitters**: strategy_engine
- **Listeners**: execution_engine

### ORDER_EXECUTED
- **Emitters**: execution_engine
- **Listeners**: position_manager

### POSITION_OPENED
- **Emitters**: position_manager
- **Listeners**: None

### POSITION_CLOSED
- **Emitters**: position_manager
- **Listeners**: None

### PRICE_FEED_UPDATE
- **Emitters**: mt5_connector
- **Listeners**: strategy_engine

### KILL_SWITCH_TRIGGERED
- **Emitters**: risk_guard
- **Listeners**: execution_engine, position_manager

### RISK_LEVEL_CHANGE
- **Emitters**: risk_guard
- **Listeners**: strategy_engine

### TICK_DATA
- **Emitters**: mt5_connector
- **Listeners**: strategy_engine, risk_guard

## 🔧 Trading Module Status

| Module | Status | Reason |
|--------|--------|--------|
| mt5_connector | ACTIVE | N/A |
| execution_engine | ACTIVE | N/A |
| strategy_engine | ACTIVE | N/A |
| position_manager | ACTIVE | N/A |
| risk_guard | ACTIVE | N/A |
| kill_switch_audit | ACTIVE | N/A |
| trader_core | Unknown | N/A |
| signal_engine | ACTIVE | N/A |

## ❌ Failed Modules

*No failed modules*
