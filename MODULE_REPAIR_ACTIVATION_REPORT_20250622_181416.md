# GENESIS MODULE REPAIR & ACTIVATION REPORT
## Generated: 2025-06-22 18:14:16

## 📊 Summary

- **Quarantined Modules**: 0
- **Repaired Modules**: 0
- **Activated Modules**: 0
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
| mt5_connector | Unknown | N/A |
| execution_engine | Unknown | N/A |
| strategy_engine | Unknown | N/A |
| position_manager | Unknown | N/A |
| risk_guard | Unknown | N/A |
| kill_switch_audit | Unknown | N/A |
| trader_core | Unknown | N/A |
| signal_engine | Unknown | N/A |

## ❌ Failed Modules

*No failed modules*
