# 🧠 GENESIS MODULES AUDIT REPORT

**Engine:** OrphanModuleRestructureEngine v1.0.0  
**Timestamp:** 2025-06-21T18:05:31.679496  
**Mission:** Full System Restructure & Integration Mode

## 📊 Summary Statistics

- **Total Modules Scanned:** 1920
- **Orphaned Modules Found:** 0
- **Modules Integrated:** 0
- **Modules Enhanced:** 0
- **Modules Flagged:** 1920
- **Modules Rejected:** 0

## ✅ Successfully Integrated Modules

| Module Name | Category | Source Path | Target Path | Status |
|-------------|----------|-------------|-------------|--------|

## 🔧 Enhanced Modules

| Module Path | Enhancements | Timestamp |
|-------------|--------------|-----------|

## ⚠️ Flagged Modules (Require Manual Review)

| Module Name | Purpose | Violations | Path |
|-------------|---------|------------|------|
| genesis_institutional_mt5_connector | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\data\genesis_institutional_mt5_connector.py |
| market_data_feed_manager | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\data\market_data_feed_manager.py |
| market_data_feed_manager_simple | DATA_PROCESSING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\data\market_data_feed_manager_simple.py |
| market_data_feed_manager_simple_integrated_2025-06-21 | DATA_PROCESSING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\data\market_data_feed_manager_simple_integrated_2025-06-21.py |
| mt5_adapter | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\data\mt5_adapter.py |
| mt5_connector_stub | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\data\mt5_connector_stub.py |
| mt5_order_executor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\data\mt5_order_executor.py |
| phase_97_1_mt5_indicator_scanner | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\data\phase_97_1_mt5_indicator_scanner.py |
| active_trades | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\execution\active_trades.py |
| adaptive_execution_resolver | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\adaptive_execution_resolver.py |
| auto_execution_manager_fixed | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\auto_execution_manager_fixed.py |
| auto_execution_manager_recovered | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\auto_execution_manager_recovered.py |
| auto_execution_sync_engine | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\auto_execution_sync_engine.py |
| byteordercodes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\byteordercodes.py |
| comprehensive_performance_testing_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\comprehensive_performance_testing_engine.py |
| contextual_execution_router | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\contextual_execution_router.py |
| dashboard_linkage_patch | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\execution\dashboard_linkage_patch.py |
| execution_engine | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\execution_engine.py |
| execution_flow_controller | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\execution_flow_controller.py |
| execution_harmonizer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\execution_harmonizer.py |
| execution_loop_responder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\execution_loop_responder.py |
| execution_manager | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\execution_manager.py |
| execution_prioritization_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\execution_prioritization_engine.py |
| execution_supervisor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\execution_supervisor.py |
| execution_supervisor_new | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\execution_supervisor_new.py |
| execution_supervisor_new_restored | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\execution_supervisor_new_restored.py |
| execution_supervisor_recovered | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\execution_supervisor_recovered.py |
| ftmo_limit_guard | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\ftmo_limit_guard.py |
| genesis_institutional_execution_middleware | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\genesis_institutional_execution_middleware.py |
| genesis_launcher | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\execution\genesis_launcher.py |
| genesis_trade_engine | TRADING | STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\genesis_trade_engine.py |
| interpolatableTestContourOrder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\interpolatableTestContourOrder.py |
| live_backtest_comparison_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\live_backtest_comparison_engine.py |
| live_feedback_adapter | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\live_feedback_adapter.py |
| mutator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\mutator.py |
| ordered | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\ordered.py |
| phase_102_kill_switch_execution_loop | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\phase_102_kill_switch_execution_loop.py |
| post_trade_feedback_engine | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\post_trade_feedback_engine.py |
| RECOVERED_execution_supervisor_ARCHITECT_COMPLIANT | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\RECOVERED_execution_supervisor_ARCHITECT_COMPLIANT.py |
| signal_execution_router_recovered_3 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\signal_execution_router_recovered_3.py |
| smart_execution_liveloop | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\execution\smart_execution_liveloop.py |
| smart_execution_liveloop_v7 | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\smart_execution_liveloop_v7.py |
| smart_execution_reactor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\smart_execution_reactor.py |
| standardGlyphOrder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\standardGlyphOrder.py |
| test_phase18_reactive_execution | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\test_phase18_reactive_execution.py |
| test_phase18_reactive_execution_fixed | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\test_phase18_reactive_execution_fixed.py |
| trade_journal | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\execution\trade_journal.py |
| universal_mt5_discovery_engine | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\execution\universal_mt5_discovery_engine.py |
| validate_phase38_execution_selector | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\validate_phase38_execution_selector.py |
| _borderpad | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\_borderpad.py |
| _borderwidth | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\_borderwidth.py |
| _categoryorder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\_categoryorder.py |
| _columnorder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\_columnorder.py |
| _columnordersrc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\_columnordersrc.py |
| _ordering | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\_ordering.py |
| _roworder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\_roworder.py |
| _traceorder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\_traceorder.py |
| _zorder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\_zorder.py |
| execution_engine_orchestrator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\orchestrator\execution_engine_orchestrator.py |
| signal_execution_router | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\router\signal_execution_router.py |
| execution_engine_v3_phase66 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\execution\v3\execution_engine_v3_phase66.py |
| genesis_institutional_risk_engine_v7 | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\institutional\genesis_institutional_risk_engine_v7.py |
| mt5_adapter_v7 | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\institutional\mt5_adapter_v7.py |
| market_data_feed_manager_simple | DATA_PROCESSING | MISSING_EVENTBUS_INTEGRATION, MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\market_data\market_data_feed_manager_simple.py |
| adaptive_filter_engine_simple | SIGNAL_PROCESSING | MISSING_EVENTBUS_INTEGRATION, MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\ml\adaptive_filter_engine_simple.py |
| advanced_pattern_miner | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\ml\advanced_pattern_miner.py |
| pattern_classifier_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\ml\pattern_classifier_engine.py |
| pattern_engine | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\ml\pattern_engine.py |
| ml_pattern_engine_v7_clean | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\ml\v7\ml_pattern_engine_v7_clean.py |
| test_portfolio_optimizer_phase47 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\ml_optimization\test_portfolio_optimizer_phase47.py |
| _hessian_update_strategy | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\ml_optimization\_hessian_update_strategy.py |
| _stochastic_optimizers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\ml_optimization\_stochastic_optimizers.py |
| final_system_validator | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\monitoring\final_system_validator.py |
| fixed_monitor_test | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\monitoring\fixed_monitor_test.py |
| lightweight_validation_hook | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\monitoring\lightweight_validation_hook.py |
| live_trade_analyzer | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\monitoring\live_trade_analyzer.py |
| performance_status_check | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\monitoring\performance_status_check.py |
| portfolio_optimizer | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\optimization\portfolio_optimizer.py |
| ml_pattern_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\patterns\ml_pattern_engine.py |
| pattern_feedback_loop_integrator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\patterns\pattern_feedback_loop_integrator.py |
| pattern_learning_engine_phase58 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\patterns\pattern_learning_engine_phase58.py |
| pattern_learning_engine_phase58_simple | TRADING | MISSING_EVENTBUS_INTEGRATION, MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\patterns\pattern_learning_engine_phase58_simple.py |
| pattern_learning_engine_v7_clean | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\patterns\pattern_learning_engine_v7_clean.py |
| abc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\abc.py |
| abstract | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\abstract.py |
| accessor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\accessor.py |
| accessors | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\accessors.py |
| accumulate | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\accumulate.py |
| acero | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\acero.py |
| actions | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\actions.py |
| activate_phase_91c_lockdown | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\activate_phase_91c_lockdown.py |
| adapter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\adapter.py |
| adapters | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\adapters.py |
| afmLib | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\afmLib.py |
| agl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\agl.py |
| alert | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\alert.py |
| algorithms | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\algorithms.py |
| align | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\align.py |
| anchored_artists | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\anchored_artists.py |
| android | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\android.py |
| angle_helper | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\angle_helper.py |
| animation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\animation.py |
| ansi | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ansi.py |
| ansitowin32 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ansitowin32.py |
| ansitowin32_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ansitowin32_test.py |
| ansi_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ansi_test.py |
| any_namespace | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\any_namespace.py |
| api | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\api.py |
| appengine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\appengine.py |
| apply | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\apply.py |
| app_session | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\app_session.py |
| app_static_file_handler | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\app_static_file_handler.py |
| app_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\app_test.py |
| arc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\arc.py |
| architect_surveillance_daemon | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\architect_surveillance_daemon.py |
| areaPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\areaPen.py |
| arithmetic | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\arithmetic.py |
| arpack | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\arpack.py |
| array | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\array.py |
| arraylike | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\arraylike.py |
| arrayprint | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\arrayprint.py |
| arrayTools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\arrayTools.py |
| array_constructors | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\array_constructors.py |
| array_manager | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\array_manager.py |
| array_ops | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\array_ops.py |
| arrow | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\arrow.py |
| arrow_parser_wrapper | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\arrow_parser_wrapper.py |
| art3d | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\art3d.py |
| artist | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\artist.py |
| asizeof | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\asizeof.py |
| asserters | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\asserters.py |
| ast | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ast.py |
| asyncio | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\asyncio.py |
| asyncio_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\asyncio_test.py |
| async_utils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\async_utils.py |
| audio_input | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\audio_input.py |
| audit_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\audit_engine.py |
| auth | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\auth.py |
| auth_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\auth_test.py |
| auth_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\auth_util.py |
| autoreload_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\autoreload_test.py |
| auxfuncs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\auxfuncs.py |
| avarPlanner | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\avarPlanner.py |
| AvifImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\AvifImagePlugin.py |
| axes3d | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\axes3d.py |
| axes_grid | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\axes_grid.py |
| axes_rgb | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\axes_rgb.py |
| axes_size | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\axes_size.py |
| axis | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\axis.py |
| axis3d | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\axis3d.py |
| axislines | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\axislines.py |
| axisline_style | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\axisline_style.py |
| axis_artist | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\axis_artist.py |
| backend_agg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_agg.py |
| backend_bases | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_bases.py |
| backend_cairo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_cairo.py |
| backend_gtk3 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_gtk3.py |
| backend_gtk3agg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_gtk3agg.py |
| backend_gtk3cairo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_gtk3cairo.py |
| backend_gtk4 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_gtk4.py |
| backend_gtk4cairo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_gtk4cairo.py |
| backend_macosx | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_macosx.py |
| backend_managers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_managers.py |
| backend_mixed | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_mixed.py |
| backend_nbagg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_nbagg.py |
| backend_pdf | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_pdf.py |
| backend_pgf | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_pgf.py |
| backend_ps | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_ps.py |
| backend_qt | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_qt.py |
| backend_qtagg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_qtagg.py |
| backend_qtcairo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_qtcairo.py |
| backend_svg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_svg.py |
| backend_tkagg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_tkagg.py |
| backend_tkcairo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_tkcairo.py |
| backend_tools | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_tools.py |
| backend_webagg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_webagg.py |
| backend_webagg_core | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_webagg_core.py |
| backend_wx | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_wx.py |
| backend_wxagg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_wxagg.py |
| backend_wxcairo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backend_wxcairo.py |
| backports | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\backports.py |
| bar | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bar.py |
| base | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\base.py |
| basedatatypes | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\basedatatypes.py |
| basePen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\basePen.py |
| basevalidators | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\basevalidators.py |
| basewidget | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\basewidget.py |
| base_command | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\base_command.py |
| base_component_registry | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\base_component_registry.py |
| base_connection | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\base_connection.py |
| base_custom_component | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\base_custom_component.py |
| base_parser | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\base_parser.py |
| bazaar | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bazaar.py |
| bccache | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bccache.py |
| bdf | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bdf.py |
| bezier | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bezier.py |
| bezierTools | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bezierTools.py |
| bindings | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bindings.py |
| binning | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\binning.py |
| BitmapGlyphMetrics | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\BitmapGlyphMetrics.py |
| Blocks | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\Blocks.py |
| BlpImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\BlpImagePlugin.py |
| BmpImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\BmpImagePlugin.py |
| bokeh_chart | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bokeh_chart.py |
| bokeh_renderer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bokeh_renderer.py |
| boolean | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\boolean.py |
| bootstrap | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bootstrap.py |
| boundsPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\boundsPen.py |
| box | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\box.py |
| boxplot | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\boxplot.py |
| bricks | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\bricks.py |
| browser_websocket_handler | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\browser_websocket_handler.py |
| buf | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\buf.py |
| buffer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\buffer.py |
| BufrStubImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\BufrStubImagePlugin.py |
| build_continuity_guard | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\build_continuity_guard.py |
| build_env | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\build_env.py |
| build_tracker_logger | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\build_tracker_logger.py |
| built_in_chart_utils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\built_in_chart_utils.py |
| button | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\button.py |
| button_group | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\button_group.py |
| cache | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cache.py |
| cached_message_replay | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cached_message_replay.py |
| cache_data_api | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cache_data_api.py |
| cache_errors | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cache_errors.py |
| cache_resource_api | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cache_resource_api.py |
| cache_storage_protocol | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cache_storage_protocol.py |
| cache_utils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cache_utils.py |
| cairoPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cairoPen.py |
| calibration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\calibration.py |
| callbacks | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\callbacks.py |
| camera_input | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\camera_input.py |
| candidates | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\candidates.py |
| canonical_constraint | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\canonical_constraint.py |
| capi_maps | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\capi_maps.py |
| caresresolver | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\caresresolver.py |
| cast | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cast.py |
| casting | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\casting.py |
| category | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\category.py |
| cbook | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cbook.py |
| cb_rules | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cb_rules.py |
| cd | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cd.py |
| cff | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cff.py |
| CFFToCFF2 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\CFFToCFF2.py |
| cfuncs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cfuncs.py |
| channels | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\channels.py |
| chat | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\chat.py |
| chebyshev | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\chebyshev.py |
| check | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\check.py |
| checkbox | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\checkbox.py |
| circlerefs_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\circlerefs_test.py |
| classifyTools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\classifyTools.py |
| class_weight | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\class_weight.py |
| cli | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cli.py |
| cloudpickle | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cloudpickle.py |
| cloudpickle_wrapper | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cloudpickle_wrapper.py |
| cm | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cm.py |
| cmap | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cmap.py |
| cmd | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cmd.py |
| cmdoptions | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cmdoptions.py |
| cocoaPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cocoaPen.py |
| code | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\code.py |
| codec | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\codec.py |
| codecs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\codecs.py |
| collections_quarantined | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\collections_quarantined.py |
| collector | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\collector.py |
| color | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\color.py |
| colorbar | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\colorbar.py |
| colorizer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\colorizer.py |
| colors | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\colors.py |
| color_picker | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\color_picker.py |
| color_triplet | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\color_triplet.py |
| color_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\color_util.py |
| column | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\column.py |
| columns | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\columns.py |
| column_config_utils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\column_config_utils.py |
| column_types | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\column_types.py |
| command_context | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\command_context.py |
| commit | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\commit.py |
| common | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\common.py |
| common_tests | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\common_tests.py |
| compare | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\compare.py |
| compatibility_tags | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\compatibility_tags.py |
| compiler | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\compiler.py |
| completion | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\completion.py |
| component_registry | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\component_registry.py |
| component_request_handler | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\component_request_handler.py |
| compressor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\compressor.py |
| compressors | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\compressors.py |
| compute | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\compute.py |
| concat | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\concat.py |
| concurrent | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\concurrent.py |
| concurrent_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\concurrent_test.py |
| config | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\config.py |
| configTools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\configTools.py |
| configuration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\configuration.py |
| config_option | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\config_option.py |
| confusion_matrix | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\confusion_matrix.py |
| connectionpool | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\connectionpool.py |
| connection_factory | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\connection_factory.py |
| console | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\console.py |
| constrain | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\constrain.py |
| construction | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\construction.py |
| constructors | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\constructors.py |
| container | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\container.py |
| ContainerIO | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ContainerIO.py |
| containers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\containers.py |
| context | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\context.py |
| contingency | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\contingency.py |
| contour | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\contour.py |
| control | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\control.py |
| controller | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\controller.py |
| convert | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\convert.py |
| converter | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\converter.py |
| cookies | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cookies.py |
| core | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\core.py |
| cpp_message | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cpp_message.py |
| crackfortran | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\crackfortran.py |
| credentials | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\credentials.py |
| criterion | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\criterion.py |
| css | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\css.py |
| csvs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\csvs.py |
| cu2qu | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cu2qu.py |
| cu2quPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cu2quPen.py |
| curl_httpclient | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\curl_httpclient.py |
| curl_httpclient_test | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\curl_httpclient_test.py |
| cursor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\cursor.py |
| custom_component | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\custom_component.py |
| C_B_D_T_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\C_B_D_T_.py |
| C_F_F_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\C_F_F_.py |
| C_O_L_R_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\C_O_L_R_.py |
| c_parser_wrapper | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\c_parser_wrapper.py |
| C_P_A_L_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\C_P_A_L_.py |
| dashboard | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dashboard.py |
| database | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\database.py |
| dataframe | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dataframe.py |
| dataframe_protocol | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dataframe_protocol.py |
| dataframe_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dataframe_util.py |
| dataset | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dataset.py |
| data_editor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\data_editor.py |
| datetimelike | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\datetimelike.py |
| datetimes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\datetimes.py |
| db | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\db.py |
| DcxImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\DcxImagePlugin.py |
| DdsImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\DdsImagePlugin.py |
| debounce | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\debounce.py |
| debug | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\debug.py |
| decision_boundary | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\decision_boundary.py |
| deck | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\deck.py |
| deck_gl_json_chart | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\deck_gl_json_chart.py |
| decoder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\decoder.py |
| decorator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\decorator.py |
| decorators | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\decorators.py |
| DefaultTable | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\DefaultTable.py |
| defchararray | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\defchararray.py |
| defmatrix | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\defmatrix.py |
| delayed_queue | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\delayed_queue.py |
| delta_generator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\delta_generator.py |
| delta_generator_singletons | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\delta_generator_singletons.py |
| dependencies | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dependencies.py |
| deprecation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\deprecation.py |
| deprecation_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\deprecation_util.py |
| describe | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\describe.py |
| descriptor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\descriptor.py |
| descriptor_database | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\descriptor_database.py |
| descriptor_pool | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\descriptor_pool.py |
| det_curve | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\det_curve.py |
| dialog | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dialog.py |
| dialog_decorator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dialog_decorator.py |
| diff | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\diff.py |
| dim2 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dim2.py |
| direct_url | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\direct_url.py |
| dirsnapshot | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dirsnapshot.py |
| discovery | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\discovery.py |
| discriminant_analysis | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\discriminant_analysis.py |
| distance | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\distance.py |
| distro | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\distro.py |
| doccer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\doccer.py |
| docscrape | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\docscrape.py |
| doc_string | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\doc_string.py |
| download | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\download.py |
| driver | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\driver.py |
| dtype | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dtype.py |
| dtypes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dtypes.py |
| dummy | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dummy.py |
| dummy_cache_storage | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dummy_cache_storage.py |
| DUPLICATE_hardlock_recovery_engine_fixed | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\DUPLICATE_hardlock_recovery_engine_fixed.py |
| DUPLICATE_indicator_scanner | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\DUPLICATE_indicator_scanner.py |
| duration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\duration.py |
| dviread | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\dviread.py |
| D_S_I_G_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\D_S_I_G_.py |
| D__e_b_g | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\D__e_b_g.py |
| einsumfunc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\einsumfunc.py |
| element_tree | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\element_tree.py |
| emergency_architect_compliance_enforcer | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\emergency_architect_compliance_enforcer.py |
| emergency_architect_compliance_fixer | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\emergency_architect_compliance_fixer.py |
| emoji | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\emoji.py |
| empty | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\empty.py |
| encoder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\encoder.py |
| encryption | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\encryption.py |
| enhanced_hardlock_recovery | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\enhanced_hardlock_recovery.py |
| enum_type_wrapper | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\enum_type_wrapper.py |
| environment | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\environment.py |
| Epoch | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\Epoch.py |
| EpochConverter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\EpochConverter.py |
| EpsImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\EpsImagePlugin.py |
| errors | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\errors.py |
| error_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\error_util.py |
| escape | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\escape.py |
| escape_test | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\escape_test.py |
| estimator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\estimator.py |
| estimator_checks | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\estimator_checks.py |
| events | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\events.py |
| event_based_path_watcher | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\event_based_path_watcher.py |
| event_bus | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\event_bus.py |
| event_debouncer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\event_debouncer.py |
| ewm | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ewm.py |
| exc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\exc.py |
| excel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\excel.py |
| exception | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\exception.py |
| exceptions | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\exceptions.py |
| execeval | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\execeval.py |
| exec_code | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\exec_code.py |
| ExifTags | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ExifTags.py |
| expanding | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\expanding.py |
| exporter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\exporter.py |
| expr | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\expr.py |
| expressions | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\expressions.py |
| expr_dt | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\expr_dt.py |
| expr_name | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\expr_name.py |
| expr_str | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\expr_str.py |
| expr_struct | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\expr_struct.py |
| ext | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ext.py |
| extension | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\extension.py |
| extensions | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\extensions.py |
| extension_dict | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\extension_dict.py |
| extension_types | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\extension_types.py |
| extmath | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\extmath.py |
| extras | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\extras.py |
| E_B_D_T_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\E_B_D_T_.py |
| E_B_L_C_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\E_B_L_C_.py |
| f2py2e | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\f2py2e.py |
| factory | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\factory.py |
| fake_renderer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fake_renderer.py |
| fallback | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fallback.py |
| feather | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\feather.py |
| features | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\features.py |
| fetch | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fetch.py |
| fields | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fields.py |
| field_mask | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\field_mask.py |
| figmpl_directive | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\figmpl_directive.py |
| figure | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\figure.py |
| filenames | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\filenames.py |
| filewrapper | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\filewrapper.py |
| file_cache | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\file_cache.py |
| file_proxy | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\file_proxy.py |
| file_uploader | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\file_uploader.py |
| file_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\file_util.py |
| filter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\filter.py |
| filterPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\filterPen.py |
| final_system_reconstructor | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\final_system_reconstructor.py |
| FitsImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\FitsImagePlugin.py |
| fixes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fixes.py |
| flags | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\flags.py |
| FliImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\FliImagePlugin.py |
| floating | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\floating.py |
| floating_axes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\floating_axes.py |
| folder_black_list | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\folder_black_list.py |
| fontBuilder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fontBuilder.py |
| FontFile | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\FontFile.py |
| font_manager | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\font_manager.py |
| form | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\form.py |
| format | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\format.py |
| formatter | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\formatter.py |
| formatting | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\formatting.py |
| format_control | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\format_control.py |
| form_utils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\form_utils.py |
| forward_msg_queue | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\forward_msg_queue.py |
| found_candidates | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\found_candidates.py |
| FpxImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\FpxImagePlugin.py |
| fragment | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fragment.py |
| frame | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\frame.py |
| framework | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\framework.py |
| freetypePen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\freetypePen.py |
| freeze | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\freeze.py |
| frequencies | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\frequencies.py |
| fromnumeric | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fromnumeric.py |
| from_dataframe | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\from_dataframe.py |
| frozen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\frozen.py |
| fs | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fs.py |
| fsevents | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fsevents.py |
| fsevents2 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fsevents2.py |
| FtexImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\FtexImagePlugin.py |
| fun | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\fun.py |
| func | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\func.py |
| func2subr | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\func2subr.py |
| function | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\function.py |
| functions | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\functions.py |
| function_base | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\function_base.py |
| func_inspect | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\func_inspect.py |
| F_F_T_M_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\F_F_T_M_.py |
| F__e_a_t | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\F__e_a_t.py |
| GbrImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\GbrImagePlugin.py |
| GdImageFile | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\GdImageFile.py |
| gen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\gen.py |
| generic | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\generic.py |
| genesis_advanced_tkinter_ui | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_advanced_tkinter_ui.py |
| genesis_architecture_status_generator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_architecture_status_generator.py |
| genesis_comprehensive_native_launcher | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_comprehensive_native_launcher.py |
| genesis_dashboard_ui_live_sync | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_dashboard_ui_live_sync.py |
| genesis_docker_desktop | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_docker_desktop.py |
| genesis_final_integration_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_final_integration_test.py |
| genesis_functional_diagnostic_dashboard | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_functional_diagnostic_dashboard.py |
| genesis_high_architecture_boot | TRADING | MOCK_DATA_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_high_architecture_boot.py |
| genesis_installer_builder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_installer_builder.py |
| genesis_integrity_auditor | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_integrity_auditor.py |
| genesis_production | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_production.py |
| genesis_production_dashboard | TRADING | MOCK_DATA_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_production_dashboard.py |
| genesis_quarantine_recovery | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_quarantine_recovery.py |
| genesis_tkinter_VIOLATED | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_tkinter_VIOLATED.py |
| genesis_ultimate_integrated_system | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_ultimate_integrated_system.py |
| genesis_ultimate_launcher | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\genesis_ultimate_launcher.py |
| gen_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\gen_test.py |
| geo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\geo.py |
| getitem | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\getitem.py |
| getlimits | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\getlimits.py |
| GifImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\GifImagePlugin.py |
| GimpGradientFile | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\GimpGradientFile.py |
| GimpPaletteFile | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\GimpPaletteFile.py |
| git | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\git.py |
| git_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\git_util.py |
| glifLib | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\glifLib.py |
| glm | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\glm.py |
| gpos | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\gpos.py |
| gradient_boosting | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\gradient_boosting.py |
| graphviz_chart | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\graphviz_chart.py |
| GribStubImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\GribStubImagePlugin.py |
| gridspec | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\gridspec.py |
| grid_finder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\grid_finder.py |
| grid_helper_curvelinear | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\grid_helper_curvelinear.py |
| groupby | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\groupby.py |
| grouper | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\grouper.py |
| grower | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\grower.py |
| G_M_A_P_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\G_M_A_P_.py |
| G_P_K_G_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\G_P_K_G_.py |
| G__l_a_t | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\G__l_a_t.py |
| G__l_o_c | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\G__l_o_c.py |
| hardlock_recovery_full_patch | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hardlock_recovery_full_patch.py |
| hash | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hash.py |
| hashes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hashes.py |
| hashing | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hashing.py |
| hashPointPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hashPointPen.py |
| hatch | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hatch.py |
| hb | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hb.py |
| hdbscan | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hdbscan.py |
| Hdf5StubImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\Hdf5StubImagePlugin.py |
| head | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\head.py |
| heading | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\heading.py |
| helpers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\helpers.py |
| hermite | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hermite.py |
| hermite_e | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hermite_e.py |
| heuristics | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\heuristics.py |
| hierarchy | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hierarchy.py |
| highlighter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\highlighter.py |
| hist | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\hist.py |
| holiday | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\holiday.py |
| html | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\html.py |
| http1connection | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\http1connection.py |
| http1connection_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\http1connection_test.py |
| httpclient | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\httpclient.py |
| httpclient_test | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\httpclient_test.py |
| httpserver | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\httpserver.py |
| httpserver_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\httpserver_test.py |
| httputil | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\httputil.py |
| httputil_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\httputil_test.py |
| IcnsImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\IcnsImagePlugin.py |
| IcoImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\IcoImagePlugin.py |
| icon_cache | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\icon_cache.py |
| idtracking | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\idtracking.py |
| iframe | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\iframe.py |
| ImageChops | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageChops.py |
| ImageCms | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageCms.py |
| ImageDraw | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageDraw.py |
| ImageDraw2 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageDraw2.py |
| ImageEnhance | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageEnhance.py |
| ImageFile | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageFile.py |
| ImageFilter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageFilter.py |
| ImageFont | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageFont.py |
| ImageMath | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageMath.py |
| ImageMorph | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageMorph.py |
| ImageOps | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageOps.py |
| ImagePalette | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImagePalette.py |
| ImageQt | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageQt.py |
| ImageSequence | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageSequence.py |
| ImageShow | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageShow.py |
| ImageStat | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageStat.py |
| ImageTk | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageTk.py |
| ImageTransform | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageTransform.py |
| ImageWin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImageWin.py |
| image_utils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\image_utils.py |
| ImImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ImImagePlugin.py |
| import_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\import_test.py |
| indenter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\indenter.py |
| index | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\index.py |
| indexing | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\indexing.py |
| index_command | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\index_command.py |
| indicator_scanner_fixed | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\indicator_scanner_fixed.py |
| inference | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\inference.py |
| info | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\info.py |
| initialise_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\initialise_test.py |
| initializers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\initializers.py |
| inotify | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\inotify.py |
| inotify_buffer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\inotify_buffer.py |
| inotify_c | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\inotify_c.py |
| inset | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\inset.py |
| inset_locator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\inset_locator.py |
| inspect | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\inspect.py |
| install | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\install.py |
| installation_report | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\installation_report.py |
| installed | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\installed.py |
| integer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\integer.py |
| interpolatable | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\interpolatable.py |
| interpolatableHelpers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\interpolatableHelpers.py |
| interpolatablePlot | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\interpolatablePlot.py |
| interpolative | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\interpolative.py |
| interval | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\interval.py |
| in_memory_cache_storage_wrapper | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\in_memory_cache_storage_wrapper.py |
| ioloop | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ioloop.py |
| ioloop_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ioloop_test.py |
| iostream | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\iostream.py |
| iostream_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\iostream_test.py |
| ipc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ipc.py |
| IptcImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\IptcImagePlugin.py |
| isatty_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\isatty_test.py |
| isoparser | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\isoparser.py |
| isotonic | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\isotonic.py |
| iterative | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\iterative.py |
| iup | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\iup.py |
| ivp | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ivp.py |
| Jpeg2KImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\Jpeg2KImagePlugin.py |
| JpegImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\JpegImagePlugin.py |
| json | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\json.py |
| jsonschema | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\jsonschema.py |
| json_format | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\json_format.py |
| json_tools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\json_tools.py |
| js_number | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\js_number.py |
| jupyter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\jupyter.py |
| jupyter_chart | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\jupyter_chart.py |
| jvm | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\jvm.py |
| kernels | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\kernels.py |
| kernel_approximation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\kernel_approximation.py |
| kernel_ridge | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\kernel_ridge.py |
| keys | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\keys.py |
| kill_switch | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\kill_switch.py |
| kqueue | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\kqueue.py |
| laguerre | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\laguerre.py |
| launch_backend_fixed | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\launch_backend_fixed.py |
| launch_backend_full | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\launch_backend_full.py |
| launch_docker_genesis | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\launch_docker_genesis.py |
| layer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\layer.py |
| layout | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\layout.py |
| layouts | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\layouts.py |
| layout_engine | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\layout_engine.py |
| layout_utils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\layout_utils.py |
| lazy | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\lazy.py |
| lazyTools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\lazyTools.py |
| lazy_wheel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\lazy_wheel.py |
| least_squares | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\least_squares.py |
| legend | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\legend.py |
| legendre | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\legendre.py |
| legend_handler | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\legend_handler.py |
| lexer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\lexer.py |
| lib | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\lib.py |
| lines | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\lines.py |
| link | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\link.py |
| linsolve | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\linsolve.py |
| list | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\list.py |
| live | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\live.py |
| live_alert_bridge | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\live_alert_bridge.py |
| live_render | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\live_render.py |
| loader | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\loader.py |
| loaders | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\loaders.py |
| lobpcg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\lobpcg.py |
| locale | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\locale.py |
| locale_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\locale_test.py |
| local_component_registry | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\local_component_registry.py |
| local_disk_cache_storage | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\local_disk_cache_storage.py |
| local_script_runner | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\local_script_runner.py |
| local_sources_watcher | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\local_sources_watcher.py |
| locators | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\locators.py |
| locks | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\locks.py |
| locks_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\locks_test.py |
| log | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\log.py |
| logging | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\logging.py |
| loggingTools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\loggingTools.py |
| log_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\log_test.py |
| loose | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\loose.py |
| loss | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\loss.py |
| low_level | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\low_level.py |
| lsoda | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\lsoda.py |
| L_T_S_H_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\L_T_S_H_.py |
| macos | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\macos.py |
| macRes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\macRes.py |
| macro_event_sync_engine_92c | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\macro_event_sync_engine_92c.py |
| macro_sync_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\macro_sync_engine.py |
| macUtils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\macUtils.py |
| magic | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\magic.py |
| main | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\main.py |
| managers | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\managers.py |
| manifest | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\manifest.py |
| map | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\map.py |
| markdown | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\markdown.py |
| markers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\markers.py |
| markup | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\markup.py |
| masked | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\masked.py |
| masked_shared | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\masked_shared.py |
| mathmpl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mathmpl.py |
| mathtext | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mathtext.py |
| md | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\md.py |
| measure | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\measure.py |
| media | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\media.py |
| media_file_handler | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\media_file_handler.py |
| media_file_manager | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\media_file_manager.py |
| media_file_storage | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\media_file_storage.py |
| mem | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mem.py |
| memmap | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\memmap.py |
| memory | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\memory.py |
| memory_media_file_storage | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\memory_media_file_storage.py |
| memory_session_storage | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\memory_session_storage.py |
| memory_uploaded_file_manager | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\memory_uploaded_file_manager.py |
| mercurial | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mercurial.py |
| merge | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\merge.py |
| merger | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\merger.py |
| message | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\message.py |
| message_factory | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\message_factory.py |
| message_listener | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\message_listener.py |
| meta | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\meta.py |
| metadata | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\metadata.py |
| metadata_routing_common | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\metadata_routing_common.py |
| metaestimators | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\metaestimators.py |
| methods | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\methods.py |
| metric | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\metric.py |
| metrics_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\metrics_util.py |
| MicImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\MicImagePlugin.py |
| minimize_trustregion_constr | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\minimize_trustregion_constr.py |
| missing | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\missing.py |
| mixins | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mixins.py |
| mlab | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mlab.py |
| ml_retraining_loop_phase57 | TRADING | STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ml_retraining_loop_phase57.py |
| mman | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mman.py |
| mock_backend | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mock_backend.py |
| models | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\models.py |
| module_recovery_engine | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\module_recovery_engine.py |
| momentsPen | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\momentsPen.py |
| MpegImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\MpegImagePlugin.py |
| mpltools | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mpltools.py |
| mpl_axes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mpl_axes.py |
| mpl_renderer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mpl_renderer.py |
| MpoImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\MpoImagePlugin.py |
| mrecords | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mrecords.py |
| MspImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\MspImagePlugin.py |
| multi | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\multi.py |
| multiarray | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\multiarray.py |
| multiclass | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\multiclass.py |
| multioutput | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\multioutput.py |
| multiselect | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\multiselect.py |
| multiVarStore | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\multiVarStore.py |
| multi_agent_coordination_engine_fixed | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\multi_agent_coordination_engine_fixed.py |
| mutable_status_container | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mutable_status_container.py |
| mypy_plugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\mypy_plugin.py |
| M_E_T_A_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\M_E_T_A_.py |
| naive_bayes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\naive_bayes.py |
| names | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\names.py |
| namespace | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\namespace.py |
| nanops | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\nanops.py |
| nap | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\nap.py |
| nativetypes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\nativetypes.py |
| ndarray_misc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ndarray_misc.py |
| netutil | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\netutil.py |
| netutil_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\netutil_test.py |
| nodes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\nodes.py |
| ntlmpool | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ntlmpool.py |
| numba_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\numba_.py |
| number_input | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\number_input.py |
| numerictypes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\numerictypes.py |
| numpy_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\numpy_.py |
| numpy_pickle | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\numpy_pickle.py |
| numpy_pickle_compat | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\numpy_pickle_compat.py |
| numpy_pickle_utils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\numpy_pickle_utils.py |
| oauth_authlib_routes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\oauth_authlib_routes.py |
| objcreator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\objcreator.py |
| objects | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\objects.py |
| object_array | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\object_array.py |
| offline | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\offline.py |
| offsetbox | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\offsetbox.py |
| oidc_mixin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\oidc_mixin.py |
| online | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\online.py |
| ops | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ops.py |
| optimize | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\optimize.py |
| optimizer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\optimizer.py |
| options | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\options.py |
| options_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\options_test.py |
| orc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\orc.py |
| orphan_intent_classifier | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\orphan_intent_classifier.py |
| orphan_recovery_integration_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\orphan_recovery_integration_engine.py |
| otBase | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\otBase.py |
| otConverters | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\otConverters.py |
| otTables | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\otTables.py |
| otTraverse | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\otTraverse.py |
| O_S_2f_2 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\O_S_2f_2.py |
| pack | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pack.py |
| package_finder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\package_finder.py |
| padding | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\padding.py |
| page | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\page.py |
| pager | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pager.py |
| pages_manager | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pages_manager.py |
| pairwise | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pairwise.py |
| palette | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\palette.py |
| pandas_compat | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pandas_compat.py |
| panel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\panel.py |
| parallel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\parallel.py |
| params | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\params.py |
| parquet | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\parquet.py |
| parser | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\parser.py |
| partial_dependence | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\partial_dependence.py |
| patches | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\patches.py |
| path | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\path.py |
| patheffects | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\patheffects.py |
| path_watcher | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\path_watcher.py |
| PcfFontFile | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\PcfFontFile.py |
| PcxImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\PcxImagePlugin.py |
| PdfParser | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\PdfParser.py |
| performance_testing_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\performance_testing_engine.py |
| perimeterPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\perimeterPen.py |
| period | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\period.py |
| phase_101_institutional_module_registry | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\phase_101_institutional_module_registry.py |
| phase_101_institutional_module_registry_clean | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\phase_101_institutional_module_registry_clean.py |
| phase_102_preparation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\phase_102_preparation.py |
| phase_90_feedback_validation | TRADING | STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\phase_90_feedback_validation.py |
| phase_92a_complete_dashboard | TRADING | MOCK_DATA_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\phase_92a_complete_dashboard.py |
| phase_94_dependency_fingerprint_verifier | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\phase_94_dependency_fingerprint_verifier.py |
| phase_95_eventbus_focused_validator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\phase_95_eventbus_focused_validator.py |
| phase_95_eventbus_validator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\phase_95_eventbus_validator.py |
| phase_97_5_architect_validator_full | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\phase_97_5_architect_validator_full.py |
| phase_99_launch_and_link | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\phase_99_launch_and_link.py |
| pickle_compat | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pickle_compat.py |
| pipeline | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pipeline.py |
| pivot | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pivot.py |
| pkg_resources | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pkg_resources.py |
| plot | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\plot.py |
| plotly_chart | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\plotly_chart.py |
| plot_directive | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\plot_directive.py |
| plugin_registry | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\plugin_registry.py |
| png | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\png.py |
| PngImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\PngImagePlugin.py |
| pointInsidePen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pointInsidePen.py |
| polar | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\polar.py |
| policies | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\policies.py |
| polling | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\polling.py |
| polling_path_watcher | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\polling_path_watcher.py |
| polynomial | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\polynomial.py |
| polyutils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\polyutils.py |
| pool | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pool.py |
| poolmanager | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\poolmanager.py |
| popen_loky_posix | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\popen_loky_posix.py |
| popen_loky_win32 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\popen_loky_win32.py |
| PpmImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\PpmImagePlugin.py |
| precision_recall_curve | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\precision_recall_curve.py |
| predictor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\predictor.py |
| prepare | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\prepare.py |
| pretty | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pretty.py |
| printing | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\printing.py |
| print_coercion_tables | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\print_coercion_tables.py |
| probe | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\probe.py |
| problem | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\problem.py |
| process | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\process.py |
| process_executor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\process_executor.py |
| process_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\process_test.py |
| progress | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\progress.py |
| progress_bar | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\progress_bar.py |
| proj3d | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\proj3d.py |
| projections | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\projections.py |
| prompt | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\prompt.py |
| properties | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\properties.py |
| protocols | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\protocols.py |
| provider | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\provider.py |
| providers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\providers.py |
| proxy_metaclass | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\proxy_metaclass.py |
| psCharStrings | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\psCharStrings.py |
| PsdImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\PsdImagePlugin.py |
| PSDraw | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\PSDraw.py |
| psLib | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\psLib.py |
| psOperators | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\psOperators.py |
| pylock | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pylock.py |
| pyopenssl | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pyopenssl.py |
| pyplot | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pyplot.py |
| pytables | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\pytables.py |
| python | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\python.py |
| python_message | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\python_message.py |
| python_parser | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\python_parser.py |
| qobjectcreator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\qobjectcreator.py |
| QoiImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\QoiImagePlugin.py |
| qtPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\qtPen.py |
| qtproxies | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\qtproxies.py |
| qu2cuPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\qu2cuPen.py |
| quartzPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\quartzPen.py |
| query_params | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\query_params.py |
| query_params_proxy | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\query_params_proxy.py |
| queue | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\queue.py |
| queues | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\queues.py |
| queues_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\queues_test.py |
| quiver | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\quiver.py |
| radau | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\radau.py |
| radio | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\radio.py |
| random_projection | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\random_projection.py |
| range | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\range.py |
| rcsetup | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\rcsetup.py |
| readers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\readers.py |
| read_directory_changes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\read_directory_changes.py |
| recfunctions | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\recfunctions.py |
| recordingPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\recordingPen.py |
| records | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\records.py |
| redis_cache | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\redis_cache.py |
| reduce | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reduce.py |
| reduction | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reduction.py |
| ref | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ref.py |
| reference | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reference.py |
| registry | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\registry.py |
| regression | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\regression.py |
| relativedelta | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\relativedelta.py |
| remote | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\remote.py |
| removeOverlaps | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\removeOverlaps.py |
| renderer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\renderer.py |
| reorderGlyphs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reorderGlyphs.py |
| report | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\report.py |
| reporter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reporter.py |
| reporters | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reporters.py |
| reportLabPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reportLabPen.py |
| repr | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\repr.py |
| request | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\request.py |
| requirements | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\requirements.py |
| req_command | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\req_command.py |
| req_file | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\req_file.py |
| req_install | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\req_install.py |
| req_set | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\req_set.py |
| req_uninstall | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\req_uninstall.py |
| resample | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\resample.py |
| reshape | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reshape.py |
| reshaping | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reshaping.py |
| resolution | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\resolution.py |
| resolver | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\resolver.py |
| resources | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\resources.py |
| resource_tracker | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\resource_tracker.py |
| results | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\results.py |
| retry | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\retry.py |
| reusable_executor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reusable_executor.py |
| reverseContourPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\reverseContourPen.py |
| rk | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\rk.py |
| roc_curve | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\roc_curve.py |
| roles | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\roles.py |
| rolling | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\rolling.py |
| root | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\root.py |
| roperator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\roperator.py |
| roundingPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\roundingPen.py |
| routes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\routes.py |
| routing | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\routing.py |
| routing_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\routing_test.py |
| rrule | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\rrule.py |
| rule | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\rule.py |
| rules | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\rules.py |
| runtests | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\runtests.py |
| runtime | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\runtime.py |
| runtime_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\runtime_util.py |
| runtime_version | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\runtime_version.py |
| safe_session_state | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\safe_session_state.py |
| sandbox | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sandbox.py |
| sankey | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sankey.py |
| sas7bdat | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sas7bdat.py |
| sasreader | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sasreader.py |
| sas_xport | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sas_xport.py |
| sbixGlyph | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sbixGlyph.py |
| sbixStrike | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sbixStrike.py |
| scalars | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\scalars.py |
| scale | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\scale.py |
| scaleUpem | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\scaleUpem.py |
| scanner | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\scanner.py |
| schema | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\schema.py |
| schemapi | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\schemapi.py |
| Scripts | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\Scripts.py |
| script_cache | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\script_cache.py |
| script_requests | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\script_requests.py |
| script_runner | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\script_runner.py |
| script_run_context | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\script_run_context.py |
| sdist | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sdist.py |
| search | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\search.py |
| search_scope | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\search_scope.py |
| secrets | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\secrets.py |
| securetransport | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\securetransport.py |
| segment | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\segment.py |
| selectbox | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\selectbox.py |
| selection | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\selection.py |
| selectn | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\selectn.py |
| selectors | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\selectors.py |
| select_slider | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\select_slider.py |
| self_outdated_check | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\self_outdated_check.py |
| serialize | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\serialize.py |
| series | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\series.py |
| series_dt | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\series_dt.py |
| series_str | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\series_str.py |
| server | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\server.py |
| service_reflection | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\service_reflection.py |
| session | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\session.py |
| sessions | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sessions.py |
| session_manager | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\session_manager.py |
| session_state | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\session_state.py |
| session_state_proxy | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\session_state_proxy.py |
| setitem | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\setitem.py |
| settings | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\settings.py |
| sfnt | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sfnt.py |
| SgiImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\SgiImagePlugin.py |
| shapeannotation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\shapeannotation.py |
| shapes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\shapes.py |
| shape_base | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\shape_base.py |
| shell_completion | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\shell_completion.py |
| show | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\show.py |
| six | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\six.py |
| slider | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\slider.py |
| smart_feedback_sync | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\smart_feedback_sync.py |
| snowflake_connection | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\snowflake_connection.py |
| snowpark_connection | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\snowpark_connection.py |
| socks | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\socks.py |
| sorting | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sorting.py |
| sources | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sources.py |
| source_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\source_util.py |
| sparsefuncs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sparsefuncs.py |
| specializer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\specializer.py |
| specifiers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\specifiers.py |
| sphinxext | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sphinxext.py |
| SpiderImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\SpiderImagePlugin.py |
| spines | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\spines.py |
| spinners | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\spinners.py |
| sql | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sql.py |
| sql_connection | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sql_connection.py |
| ssltransport | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ssltransport.py |
| ssl_ | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ssl_.py |
| sstruct | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\sstruct.py |
| stata | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\stata.py |
| statisticsPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\statisticsPen.py |
| statNames | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\statNames.py |
| stats | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\stats.py |
| stats_request_handler | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\stats_request_handler.py |
| status | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\status.py |
| step7_comprehensive_smart_monitor_validator | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\step7_comprehensive_smart_monitor_validator.py |
| stop | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\stop.py |
| strategies | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\strategies.py |
| StrConverter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\StrConverter.py |
| stream | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\stream.py |
| streamlit_callback_handler | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\streamlit_callback_handler.py |
| streamplot | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\streamplot.py |
| string | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\string.py |
| strings | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\strings.py |
| string_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\string_.py |
| string_arrow | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\string_arrow.py |
| string_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\string_util.py |
| structs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\structs.py |
| structures | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\structures.py |
| style | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\style.py |
| styled | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\styled.py |
| style_render | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\style_render.py |
| subversion | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\subversion.py |
| SunImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\SunImagePlugin.py |
| svg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\svg.py |
| svgPathPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\svgPathPen.py |
| symbolic | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\symbolic.py |
| symbol_database | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\symbol_database.py |
| symfont | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\symfont.py |
| synchronize | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\synchronize.py |
| syntax | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\syntax.py |
| system_monitor_visualizer | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\system_monitor_visualizer.py |
| system_tree_initializer | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\system_tree_initializer.py |
| system_tree_rebuild_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\system_tree_rebuild_engine.py |
| S_I_N_G_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\S_I_N_G_.py |
| S_V_G_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\S_V_G_.py |
| S__i_l_f | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\S__i_l_f.py |
| S__i_l_l | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\S__i_l_l.py |
| t2CharStringPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\t2CharStringPen.py |
| table | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\table.py |
| table_builder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\table_builder.py |
| tag | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tag.py |
| tags | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tags.py |
| take | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\take.py |
| target_python | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\target_python.py |
| tcpclient | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tcpclient.py |
| tcpclient_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tcpclient_test.py |
| tcpserver | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tcpserver.py |
| tcpserver_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tcpserver_test.py |
| teePen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\teePen.py |
| termui | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\termui.py |
| testing | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\testing.py |
| testing_refleaks | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\testing_refleaks.py |
| testing_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\testing_test.py |
| tests | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tests.py |
| testTools | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\testTools.py |
| testutils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\testutils.py |
| test_linprog | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\test_linprog.py |
| texmanager | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\texmanager.py |
| textpath | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\textpath.py |
| textTools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\textTools.py |
| text_format | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\text_format.py |
| text_widgets | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\text_widgets.py |
| tfmLib | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tfmLib.py |
| TgaImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\TgaImagePlugin.py |
| theme | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\theme.py |
| threadpoolctl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\threadpoolctl.py |
| ticker | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ticker.py |
| TiffImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\TiffImagePlugin.py |
| TiffTags | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\TiffTags.py |
| tile | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tile.py |
| timeout | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\timeout.py |
| timeseries | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\timeseries.py |
| timestamp | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\timestamp.py |
| time_widgets | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\time_widgets.py |
| toast | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\toast.py |
| token | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\token.py |
| traceback | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\traceback.py |
| transform | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\transform.py |
| transformPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\transformPen.py |
| translate | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\translate.py |
| tree | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tree.py |
| tr_interior_point | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tr_interior_point.py |
| ttCollection | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ttCollection.py |
| ttFont | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ttFont.py |
| ttGlyphPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ttGlyphPen.py |
| ttGlyphSet | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ttGlyphSet.py |
| ttProgram | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ttProgram.py |
| ttVisitor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ttVisitor.py |
| ttx | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ttx.py |
| TupleVariation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\TupleVariation.py |
| twisted_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\twisted_test.py |
| types | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\types.py |
| type_checkers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\type_checkers.py |
| type_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\type_util.py |
| typing | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\typing.py |
| typing_extensions | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\typing_extensions.py |
| tz | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tz.py |
| tzinfo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\tzinfo.py |
| T_S_I__0 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\T_S_I__0.py |
| T_S_I__1 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\T_S_I__1.py |
| T_S_I__5 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\T_S_I__5.py |
| ufo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ufo.py |
| ufunclike | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ufunclike.py |
| ufunc_config | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ufunc_config.py |
| uiparser | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\uiparser.py |
| ultimate_api | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ultimate_api.py |
| ultra_simple_auto_repair | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\ultra_simple_auto_repair.py |
| unicode | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\unicode.py |
| UnitDbl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\UnitDbl.py |
| UnitDblConverter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\UnitDblConverter.py |
| UnitDblFormatter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\UnitDblFormatter.py |
| units | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\units.py |
| unix | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\unix.py |
| unknown_fields | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\unknown_fields.py |
| unpacking | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\unpacking.py |
| uploaded_file_manager | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\uploaded_file_manager.py |
| upload_file_request_handler | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\upload_file_request_handler.py |
| url | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\url.py |
| user_info | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\user_info.py |
| utils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\utils.py |
| util_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\util_test.py |
| uts46data | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\uts46data.py |
| validate_phase23 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\validate_phase23.py |
| validate_phase45_integration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\validate_phase45_integration.py |
| validate_phase45_integration_fixed | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\validate_phase45_integration_fixed.py |
| validation | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\validation.py |
| validators | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\validators.py |
| variableScalar | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\variableScalar.py |
| varStore | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\varStore.py |
| vector | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\vector.py |
| vega_charts | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\vega_charts.py |
| vega_renderer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\vega_renderer.py |
| versioncontrol | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\versioncontrol.py |
| view | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\view.py |
| vincent_renderer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\vincent_renderer.py |
| visitor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\visitor.py |
| voltToFea | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\voltToFea.py |
| vq | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\vq.py |
| V_D_M_X_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\V_D_M_X_.py |
| V_O_R_G_ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\V_O_R_G_.py |
| WalImageFile | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\WalImageFile.py |
| watchdog_core | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\watchdog_core.py |
| watchmedo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\watchmedo.py |
| wavfile | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\wavfile.py |
| weakref_finalize | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\weakref_finalize.py |
| web | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\web.py |
| WebPImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\WebPImagePlugin.py |
| websocket | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\websocket.py |
| websocket_session_manager | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\websocket_session_manager.py |
| websocket_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\websocket_test.py |
| web_test | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\web_test.py |
| well_known_types | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\well_known_types.py |
| wheel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\wheel.py |
| when_then | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\when_then.py |
| widget | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\widget.py |
| width | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\width.py |
| win | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\win.py |
| win32 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\win32.py |
| winapi | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\winapi.py |
| winterm | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\winterm.py |
| winterm_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\winterm_test.py |
| wire_format | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\wire_format.py |
| WmfImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\WmfImagePlugin.py |
| woff2 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\woff2.py |
| wright_bessel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\wright_bessel.py |
| write | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\write.py |
| wsgi | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\wsgi.py |
| wsgi_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\wsgi_test.py |
| wxPen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\wxPen.py |
| XbmImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\XbmImagePlugin.py |
| xml | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\xml.py |
| xmlReader | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\xmlReader.py |
| xmlWriter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\xmlWriter.py |
| XpmImagePlugin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\XpmImagePlugin.py |
| _add_newdocs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_add_newdocs.py |
| _affinity_propagation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_affinity_propagation.py |
| _afm | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_afm.py |
| _agglomerative | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_agglomerative.py |
| _aliases | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_aliases.py |
| _annotated_heatmap | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_annotated_heatmap.py |
| _annotation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_annotation.py |
| _api | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_api.py |
| _arff | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_arff.py |
| _arffread | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_arffread.py |
| _arraypad_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_arraypad_impl.py |
| _arraysetops_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_arraysetops_impl.py |
| _arrayterator_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_arrayterator_impl.py |
| _array_api | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_array_api.py |
| _array_api_info | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_array_api_info.py |
| _array_like | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_array_like.py |
| _arrow_string_mixins | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_arrow_string_mixins.py |
| _at | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_at.py |
| _attrs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_attrs.py |
| _available_if | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_available_if.py |
| _axes | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_axes.py |
| _axis_nan_policy | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_axis_nan_policy.py |
| _a_v_a_r | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_a_v_a_r.py |
| _backend | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_backend.py |
| _backends | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_backends.py |
| _backend_gtk | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_backend_gtk.py |
| _backend_pdf_ps | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_backend_pdf_ps.py |
| _backend_tk | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_backend_tk.py |
| _bagging | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_bagging.py |
| _bary_rational | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_bary_rational.py |
| _base | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_base.py |
| _base_connection | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_base_connection.py |
| _base_renderers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_base_renderers.py |
| _basic | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_basic.py |
| _basic_backend | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_basic_backend.py |
| _basinhopping | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_basinhopping.py |
| _bayes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_bayes.py |
| _bayesian_mixture | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_bayesian_mixture.py |
| _binary | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_binary.py |
| _binomtest | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_binomtest.py |
| _birch | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_birch.py |
| _bisect_k_means | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_bisect_k_means.py |
| _bracket | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_bracket.py |
| _bsplines | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_bsplines.py |
| _bsr | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_bsr.py |
| _bunch | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_bunch.py |
| _button | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_button.py |
| _bvp | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_bvp.py |
| _cached | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_cached.py |
| _cachedmethod | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_cachedmethod.py |
| _calamine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_calamine.py |
| _ccallback | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ccallback.py |
| _censored_data | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_censored_data.py |
| _chandrupatla | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_chandrupatla.py |
| _chart_types | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_chart_types.py |
| _classes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_classes.py |
| _classification | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_classification.py |
| _classification_threshold | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_classification_threshold.py |
| _cm | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_cm.py |
| _cobyla_py | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_cobyla_py.py |
| _codata | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_codata.py |
| _collections | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_collections.py |
| _column_transformer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_column_transformer.py |
| _complex | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_complex.py |
| _compressed | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_compressed.py |
| _constrained_layout | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_constrained_layout.py |
| _constraints | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_constraints.py |
| _construct | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_construct.py |
| _continuous_distns | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_continuous_distns.py |
| _coo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_coo.py |
| _coordinate_descent | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_coordinate_descent.py |
| _core | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_core.py |
| _covariance | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_covariance.py |
| _csc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_csc.py |
| _csr | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_csr.py |
| _ctypeslib | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ctypeslib.py |
| _cubature | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_cubature.py |
| _cubic | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_cubic.py |
| _czt | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_czt.py |
| _c_m_a_p | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_c_m_a_p.py |
| _c_v_a_r | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_c_v_a_r.py |
| _c_v_t | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_c_v_t.py |
| _dask | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dask.py |
| _data | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_data.py |
| _datasource | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_datasource.py |
| _dbscan | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dbscan.py |
| _dcsrch | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dcsrch.py |
| _decomp | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_decomp.py |
| _decorators | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_decorators.py |
| _delegators | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_delegators.py |
| _dendrogram | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dendrogram.py |
| _deprecations | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_deprecations.py |
| _dfi_types | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dfi_types.py |
| _dia | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dia.py |
| _dict_learning | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dict_learning.py |
| _dict_vectorizer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dict_vectorizer.py |
| _differentiable_functions | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_differentiable_functions.py |
| _differentialevolution | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_differentialevolution.py |
| _differentiate | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_differentiate.py |
| _dimension | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dimension.py |
| _discrete_distns | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_discrete_distns.py |
| _discretization | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_discretization.py |
| _disjoint_set | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_disjoint_set.py |
| _dispatcher | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dispatcher.py |
| _distn_infrastructure | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_distn_infrastructure.py |
| _distplot | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_distplot.py |
| _distribution_infrastructure | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_distribution_infrastructure.py |
| _dists | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dists.py |
| _docscrape | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_docscrape.py |
| _docstring | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_docstring.py |
| _doctools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_doctools.py |
| _dok | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dok.py |
| _dtype | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dtype.py |
| _dtypes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dtypes.py |
| _dtype_ctypes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dtype_ctypes.py |
| _dtype_like | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dtype_like.py |
| _dual_annealing | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_dual_annealing.py |
| _elementwise | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_elementwise.py |
| _elffile | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_elffile.py |
| _elliptic_envelope | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_elliptic_envelope.py |
| _empirical_covariance | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_empirical_covariance.py |
| _encode | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_encode.py |
| _encoders | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_encoders.py |
| _entropy | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_entropy.py |
| _enum | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_enum.py |
| _enums | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_enums.py |
| _envs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_envs.py |
| _expm_frechet | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_expm_frechet.py |
| _expm_multiply | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_expm_multiply.py |
| _export | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_export.py |
| _expression_parsing | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_expression_parsing.py |
| _facet_grid | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_facet_grid.py |
| _factories | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_factories.py |
| _factor_analysis | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_factor_analysis.py |
| _fastica | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_fastica.py |
| _feature_agglomeration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_feature_agglomeration.py |
| _fft | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_fft.py |
| _figure | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_figure.py |
| _figurewidget | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_figurewidget.py |
| _filters | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_filters.py |
| _filter_design | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_filter_design.py |
| _fir_filter_design | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_fir_filter_design.py |
| _fit | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_fit.py |
| _fitpack2 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_fitpack2.py |
| _fitpack_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_fitpack_impl.py |
| _fitpack_py | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_fitpack_py.py |
| _fitpack_repro | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_fitpack_repro.py |
| _forest | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_forest.py |
| _format_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_format_impl.py |
| _formlayout | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_formlayout.py |
| _fortran | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_fortran.py |
| _fortran_format_parser | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_fortran_format_parser.py |
| _frame | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_frame.py |
| _from_model | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_from_model.py |
| _frozen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_frozen.py |
| _funcs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_funcs.py |
| _function_base_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_function_base_impl.py |
| _function_transformer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_function_transformer.py |
| _f_p_g_m | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_f_p_g_m.py |
| _f_v_a_r | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_f_v_a_r.py |
| _gaussian_mixture | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_gaussian_mixture.py |
| _gauss_kronrod | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_gauss_kronrod.py |
| _gauss_legendre | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_gauss_legendre.py |
| _gb | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_gb.py |
| _gcutils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_gcutils.py |
| _genz_malik | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_genz_malik.py |
| _globals | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_globals.py |
| _gpc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_gpc.py |
| _gpr | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_gpr.py |
| _graph | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_graph.py |
| _graph_lasso | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_graph_lasso.py |
| _g_a_s_p | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_g_a_s_p.py |
| _g_l_y_f | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_g_l_y_f.py |
| _g_v_a_r | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_g_v_a_r.py |
| _hash | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_hash.py |
| _histograms_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_histograms_impl.py |
| _huber | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_huber.py |
| _hypotests | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_hypotests.py |
| _h_d_m_x | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_h_d_m_x.py |
| _h_e_a_d | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_h_e_a_d.py |
| _h_h_e_a | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_h_h_e_a.py |
| _h_m_t_x | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_h_m_t_x.py |
| _idl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_idl.py |
| _iforest | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_iforest.py |
| _impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_impl.py |
| _implementation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_implementation.py |
| _incremental_pca | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_incremental_pca.py |
| _index | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_index.py |
| _indexing | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_indexing.py |
| _index_tricks_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_index_tricks_impl.py |
| _info | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_info.py |
| _inspect | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_inspect.py |
| _interface | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_interface.py |
| _interpolate | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_interpolate.py |
| _interpolation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_interpolation.py |
| _in_process | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_in_process.py |
| _iotools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_iotools.py |
| _isomap | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_isomap.py |
| _iterative | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_iterative.py |
| _json | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_json.py |
| _kaleido | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_kaleido.py |
| _kde | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_kde.py |
| _kdtree | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_kdtree.py |
| _kernel_pca | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_kernel_pca.py |
| _keywords | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_keywords.py |
| _kmeans | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_kmeans.py |
| _knn | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_knn.py |
| _ksstats | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ksstats.py |
| _k_e_r_n | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_k_e_r_n.py |
| _label | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_label.py |
| _label_propagation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_label_propagation.py |
| _laplacian | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_laplacian.py |
| _layoutgrid | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_layoutgrid.py |
| _lbfgsb_py | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_lbfgsb_py.py |
| _lda | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_lda.py |
| _least_angle | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_least_angle.py |
| _lebedev | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_lebedev.py |
| _lil | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_lil.py |
| _linalg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_linalg.py |
| _linear_loss | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_linear_loss.py |
| _linesearch | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_linesearch.py |
| _linprog_ip | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_linprog_ip.py |
| _linprog_rs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_linprog_rs.py |
| _linprog_util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_linprog_util.py |
| _locales | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_locales.py |
| _locally_linear | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_locally_linear.py |
| _lof | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_lof.py |
| _log | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_log.py |
| _logistic | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_logistic.py |
| _ltisys | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ltisys.py |
| _lti_conversion | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_lti_conversion.py |
| _l_o_c_a | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_l_o_c_a.py |
| _l_t_a_g | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_l_t_a_g.py |
| _machar | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_machar.py |
| _macos | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_macos.py |
| _make | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_make.py |
| _mannwhitneyu | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mannwhitneyu.py |
| _manylinux | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_manylinux.py |
| _matfuncs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_matfuncs.py |
| _matfuncs_inv_ssq | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_matfuncs_inv_ssq.py |
| _mathtext | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mathtext.py |
| _matrix | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_matrix.py |
| _mds | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mds.py |
| _mean_shift | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mean_shift.py |
| _measurements | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_measurements.py |
| _memmapping_reducer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_memmapping_reducer.py |
| _meson | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_meson.py |
| _metadata_requests | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_metadata_requests.py |
| _methods | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_methods.py |
| _mgc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mgc.py |
| _minimize | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_minimize.py |
| _mini_sequence_kernel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mini_sequence_kernel.py |
| _minpack_py | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_minpack_py.py |
| _mio4 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mio4.py |
| _mio5 | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mio5.py |
| _mio5_params | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mio5_params.py |
| _miobase | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_miobase.py |
| _mixins | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mixins.py |
| _mmio | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mmio.py |
| _mocking | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mocking.py |
| _models | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_models.py |
| _morestats | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_morestats.py |
| _morphology | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_morphology.py |
| _mptestutils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mptestutils.py |
| _mstats_basic | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mstats_basic.py |
| _mstats_extras | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_mstats_extras.py |
| _multicomp | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_multicomp.py |
| _multilayer_perceptron | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_multilayer_perceptron.py |
| _multiufuncs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_multiufuncs.py |
| _multivariate | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_multivariate.py |
| _musllinux | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_musllinux.py |
| _m_a_x_p | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_m_a_x_p.py |
| _m_e_t_a | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_m_e_t_a.py |
| _nanfunctions_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_nanfunctions_impl.py |
| _natype | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_natype.py |
| _nbit_base | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_nbit_base.py |
| _nca | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_nca.py |
| _ndbspline | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ndbspline.py |
| _ndgriddata | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ndgriddata.py |
| _nearest_centroid | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_nearest_centroid.py |
| _nested_sequence | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_nested_sequence.py |
| _netcdf | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_netcdf.py |
| _newton_solver | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_newton_solver.py |
| _new_distributions | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_new_distributions.py |
| _next_gen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_next_gen.py |
| _nmf | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_nmf.py |
| _nonlin | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_nonlin.py |
| _npyio_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_npyio_impl.py |
| _null_file | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_null_file.py |
| _numdiff | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_numdiff.py |
| _n_a_m_e | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_n_a_m_e.py |
| _odds_ratio | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_odds_ratio.py |
| _ode | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ode.py |
| _odepack_py | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_odepack_py.py |
| _odfreader | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_odfreader.py |
| _odrpack | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_odrpack.py |
| _odswriter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_odswriter.py |
| _omp | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_omp.py |
| _onenormest | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_onenormest.py |
| _openml | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_openml.py |
| _openpyxl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_openpyxl.py |
| _optics | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_optics.py |
| _optimize | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_optimize.py |
| _orca | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_orca.py |
| _orthogonal | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_orthogonal.py |
| _page_trend_test | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_page_trend_test.py |
| _parallel_backends | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_parallel_backends.py |
| _param_validation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_param_validation.py |
| _parser | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_parser.py |
| _partial_dependence | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_partial_dependence.py |
| _passive_aggressive | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_passive_aggressive.py |
| _pca | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pca.py |
| _peak_finding | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_peak_finding.py |
| _pep440 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pep440.py |
| _perceptron | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_perceptron.py |
| _plot | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_plot.py |
| _plotting | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_plotting.py |
| _pls | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pls.py |
| _pocketfft | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pocketfft.py |
| _polybase | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_polybase.py |
| _polyint | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_polyint.py |
| _polynomial | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_polynomial.py |
| _polynomial_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_polynomial_impl.py |
| _pprint | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pprint.py |
| _probability_distribution | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_probability_distribution.py |
| _psaix | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_psaix.py |
| _psbsd | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_psbsd.py |
| _pseudo_diffs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pseudo_diffs.py |
| _pslinux | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pslinux.py |
| _psosx | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_psosx.py |
| _pssunos | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pssunos.py |
| _pswindows | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pswindows.py |
| _pylab_helpers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pylab_helpers.py |
| _pytesttester | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pytesttester.py |
| _pyxlsb | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_pyxlsb.py |
| _p_o_s_t | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_p_o_s_t.py |
| _qap | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_qap.py |
| _qmc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_qmc.py |
| _qmvnt | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_qmvnt.py |
| _quadpack_py | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_quadpack_py.py |
| _quadrature | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_quadrature.py |
| _quad_vec | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_quad_vec.py |
| _quantile | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_quantile.py |
| _rangebreak | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_rangebreak.py |
| _ranking | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ranking.py |
| _ratio | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ratio.py |
| _rbf | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_rbf.py |
| _rbfinterp | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_rbfinterp.py |
| _rbm | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_rbm.py |
| _regression | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_regression.py |
| _renderers | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_renderers.py |
| _request_methods | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_request_methods.py |
| _resampling | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_resampling.py |
| _response | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_response.py |
| _rfe | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_rfe.py |
| _rgi | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_rgi.py |
| _ridge | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ridge.py |
| _robust_covariance | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_robust_covariance.py |
| _root | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_root.py |
| _root_scalar | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_root_scalar.py |
| _rotation_spline | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_rotation_spline.py |
| _samples_generator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_samples_generator.py |
| _sampling | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_sampling.py |
| _scimath_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_scimath_impl.py |
| _scipy_spectral_test_shim | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_scipy_spectral_test_shim.py |
| _scorer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_scorer.py |
| _search | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_search.py |
| _search_successive_halving | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_search_successive_halving.py |
| _secondary_axes | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_secondary_axes.py |
| _selection | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_selection.py |
| _self_training | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_self_training.py |
| _sensitivity_analysis | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_sensitivity_analysis.py |
| _sequential | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_sequential.py |
| _set_output | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_set_output.py |
| _sf_error | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_sf_error.py |
| _shape_base_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_shape_base_impl.py |
| _shgo | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_shgo.py |
| _short_time_fft | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_short_time_fft.py |
| _shrunk_covariance | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_shrunk_covariance.py |
| _slider | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_slider.py |
| _solvers | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_solvers.py |
| _sparse_pca | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_sparse_pca.py |
| _spdx | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_spdx.py |
| _special_inputs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_special_inputs.py |
| _special_matrices | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_special_matrices.py |
| _special_sparse_arrays | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_special_sparse_arrays.py |
| _spectral | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_spectral.py |
| _spectral_embedding | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_spectral_embedding.py |
| _spectral_py | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_spectral_py.py |
| _spherical_bessel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_spherical_bessel.py |
| _spherical_voronoi | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_spherical_voronoi.py |
| _spline_filters | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_spline_filters.py |
| _split | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_split.py |
| _sputils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_sputils.py |
| _src_pyf | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_src_pyf.py |
| _stacking | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_stacking.py |
| _stats_py | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_stats_py.py |
| _stochastic_gradient | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_stochastic_gradient.py |
| _store_backends | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_store_backends.py |
| _streamline | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_streamline.py |
| _stride_tricks_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_stride_tricks_impl.py |
| _structures | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_structures.py |
| _suite | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_suite.py |
| _supervised | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_supervised.py |
| _support_alternative_backends | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_support_alternative_backends.py |
| _survival | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_survival.py |
| _svdp | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_svdp.py |
| _s_b_i_x | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_s_b_i_x.py |
| _tags | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_tags.py |
| _tanhsinh | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_tanhsinh.py |
| _target | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_target.py |
| _target_encoder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_target_encoder.py |
| _termui_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_termui_impl.py |
| _ternary_contour | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ternary_contour.py |
| _testing | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_testing.py |
| _testutils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_testutils.py |
| _textwrap | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_textwrap.py |
| _theil_sen | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_theil_sen.py |
| _threadsafety | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_threadsafety.py |
| _tickformatstop | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_tickformatstop.py |
| _tokenizer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_tokenizer.py |
| _transformed_data | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_transformed_data.py |
| _translate | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_translate.py |
| _triangulation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_triangulation.py |
| _tricontour | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_tricontour.py |
| _trifinder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_trifinder.py |
| _triinterpolate | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_triinterpolate.py |
| _trirefine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_trirefine.py |
| _tritools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_tritools.py |
| _truncated_svd | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_truncated_svd.py |
| _trustregion | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_trustregion.py |
| _trustregion_dogleg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_trustregion_dogleg.py |
| _trustregion_exact | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_trustregion_exact.py |
| _tstutils | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_tstutils.py |
| _twodim_base_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_twodim_base_impl.py |
| _type1font | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_type1font.py |
| _type_check_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_type_check_impl.py |
| _t_r_a_k | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_t_r_a_k.py |
| _t_sne | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_t_sne.py |
| _ufunc_config | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_ufunc_config.py |
| _univariate_selection | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_univariate_selection.py |
| _updatemenu | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_updatemenu.py |
| _upfirdn | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_upfirdn.py |
| _user_array_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_user_array_impl.py |
| _util | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_util.py |
| _utilities | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_utilities.py |
| _utils_impl | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_utils_impl.py |
| _validation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_validation.py |
| _validators | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_validators.py |
| _variance_threshold | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_variance_threshold.py |
| _vegafusion_data | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_vegafusion_data.py |
| _version_info | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_version_info.py |
| _vertex | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_vertex.py |
| _voting | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_voting.py |
| _v_h_e_a | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_v_h_e_a.py |
| _warnings_errors | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_warnings_errors.py |
| _weight_boosting | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_weight_boosting.py |
| _wilcoxon | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_wilcoxon.py |
| _win32_console | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_win32_console.py |
| _winconsole | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_winconsole.py |
| _windows | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_windows.py |
| _writer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_writer.py |
| _xlrd | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_xlrd.py |
| _xlsxwriter | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_xlsxwriter.py |
| _zeros_py | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\_zeros_py.py |
| __config__ | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\restored\__config__.py |
| risk_engine | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\risk\risk_engine.py |
| _relative_risk | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\risk\_relative_risk.py |
| live_risk_governor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\risk\live\live_risk_governor.py |
| execution_risk_sentinel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\risk_management\execution_risk_sentinel.py |
| genesis_compliance_core | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\risk_management\genesis_compliance_core.py |
| genesis_institutional_risk_engine | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\risk_management\genesis_institutional_risk_engine.py |
| kill_switch_compliance | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\risk_management\kill_switch_compliance.py |
| kill_switch_logic | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\risk_management\kill_switch_logic.py |
| live_risk_governor_simple | TRADING | MISSING_EVENTBUS_INTEGRATION, MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\risk_management\live_risk_governor_simple.py |
| test_market_data_feed_manager | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\risk_management\test_market_data_feed_manager.py |
| advanced_signal_optimization_engine | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\signals\advanced_signal_optimization_engine.py |
| genesis_institutional_signal_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\genesis_institutional_signal_engine.py |
| genesis_institutional_signal_engine_v7_clean | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\genesis_institutional_signal_engine_v7_clean.py |
| institutional_signal_validator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\institutional_signal_validator.py |
| meta_signal_harmonizer | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\meta_signal_harmonizer.py |
| meta_signal_harmonizer_recovered_3 | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\meta_signal_harmonizer_recovered_3.py |
| meta_signal_harmonizer_simple | SIGNAL_PROCESSING | MISSING_EVENTBUS_INTEGRATION, MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\signals\meta_signal_harmonizer_simple.py |
| ml_execution_signal_loop | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\ml_execution_signal_loop.py |
| ORPHAN_test_advanced_signal_optimization_engine | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\ORPHAN_test_advanced_signal_optimization_engine.py |
| ORPHAN_test_signal_fusion_matrix | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\ORPHAN_test_signal_fusion_matrix.py |
| pattern_signal_harmonizer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\pattern_signal_harmonizer.py |
| phase20_signal_pipeline_stress_tester | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\phase20_signal_pipeline_stress_tester.py |
| phase_96_signal_wiring_autofix | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\phase_96_signal_wiring_autofix.py |
| phase_96_signal_wiring_enforcer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\phase_96_signal_wiring_enforcer.py |
| signal_context_enricher | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\signal_context_enricher.py |
| signal_engine | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\signal_engine.py |
| signal_fusion_matrix | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\signal_fusion_matrix.py |
| signal_historical_telemetry_linker | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\signal_historical_telemetry_linker.py |
| signal_loop_reinforcement_engine | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\signals\signal_loop_reinforcement_engine.py |
| signal_quality_amplifier | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\signal_quality_amplifier.py |
| strategic_signal_orchestrator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\strategic_signal_orchestrator.py |
| test_phase19_adaptive_signal_flow | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\test_phase19_adaptive_signal_flow.py |
| test_signaltools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\test_signaltools.py |
| _signaltools | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signals\_signaltools.py |
| adaptive_filter_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\adaptive_filter_engine.py |
| adaptive_filter_engine_simple | SIGNAL_PROCESSING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\adaptive_filter_engine_simple.py |
| adaptive_filter_engine_simple_integrated_2025-06-21 | SIGNAL_PROCESSING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\adaptive_filter_engine_simple_integrated_2025-06-21.py |
| dashboard_widgets | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\dashboard_widgets.py |
| dsr_strategy_mutator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\dsr_strategy_mutator.py |
| execution_feedback_mutator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\execution_feedback_mutator.py |
| genesis_duplicate_verification | SIGNAL_PROCESSING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\genesis_duplicate_verification.py |
| genesis_duplicate_verification_integrated_2025-06-21 | SIGNAL_PROCESSING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\genesis_duplicate_verification_integrated_2025-06-21.py |
| macro_disqualifier | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\macro_disqualifier.py |
| meta_signal_harmonizer_simple | SIGNAL_PROCESSING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\meta_signal_harmonizer_simple.py |
| meta_signal_harmonizer_simple_integrated_2025-06-21 | SIGNAL_PROCESSING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\meta_signal_harmonizer_simple_integrated_2025-06-21.py |
| mutation_signal_adapter | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\mutation_signal_adapter.py |
| patterns | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\patterns.py |
| pattern_aggregator_engine_recovered_3 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\pattern_aggregator_engine_recovered_3.py |
| pattern_confidence_overlay | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\pattern_confidence_overlay.py |
| pattern_learning_engine_phase58_recovered_3 | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\pattern_learning_engine_phase58_recovered_3.py |
| pattern_meta_strategy_engine | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\pattern_meta_strategy_engine.py |
| phase_96_signal_wiring_focused_validator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\phase_96_signal_wiring_focused_validator.py |
| signaltools | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\signaltools.py |
| signal_feed | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\signal_feed.py |
| signal_feed_generator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\signal_feed_generator.py |
| smart_signal_execution_linker | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\smart_signal_execution_linker.py |
| sniper_signal_interceptor | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\sniper_signal_interceptor.py |
| sniper_signal_interceptor_v7 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\sniper_signal_interceptor_v7.py |
| strategy_mutator | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\strategy_mutator.py |
| telemetry_dashboard | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\telemetry_dashboard.py |
| test_pattern_learning_engine_phase58 | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\test_pattern_learning_engine_phase58.py |
| test_reactive_signal_autopilot | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\test_reactive_signal_autopilot.py |
| trade_priority_resolver | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\trade_priority_resolver.py |
| validate_meta_signal_harmonizer | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\validate_meta_signal_harmonizer.py |
| validate_signalengine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\validate_signalengine.py |
| validate_signal_fusion_matrix | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\validate_signal_fusion_matrix.py |
| _fillpattern | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\_fillpattern.py |
| _pattern | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\signal_processing\_pattern.py |
| DUPLICATE_strategy_adaptive_context_synthesizer | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\strategies\DUPLICATE_strategy_adaptive_context_synthesizer.py |
| strategy_adaptive_context_synthesizer_fixed | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\strategies\strategy_adaptive_context_synthesizer_fixed.py |
| strategy_adaptive_context_synthesizer_recovered | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\strategies\strategy_adaptive_context_synthesizer_recovered.py |
| strategy_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\strategies\strategy_engine.py |
| strategy_recalibration_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\strategies\strategy_recalibration_engine.py |
| strategy_engine_fixed | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\strategies\enhanced\strategy_engine_fixed.py |
| strategy_engine_v7_clean | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\strategies\v7\strategy_engine_v7_clean.py |
| strategy_engine_v7 | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\strategy\strategy_engine_v7.py |
| strategy_recalibration_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\strategy\strategy_recalibration_test.py |
| trade_recommendation_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\strategy\trade_recommendation_engine.py |
| live_risk_governor_simple | TRADING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\trading\live_risk_governor_simple.py |
| live_risk_governor_simple_integrated_2025-06-21 | TRADING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\trading\live_risk_governor_simple_integrated_2025-06-21.py |
| mt5_order_executor_simple | TRADING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\trading\mt5_order_executor_simple.py |
| pattern_learning_engine_phase58_simple | TRADING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\trading\pattern_learning_engine_phase58_simple.py |
| pattern_learning_engine_phase58_simple_integrated_2025-06-21 | TRADING | MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\modules\trading\pattern_learning_engine_phase58_simple_integrated_2025-06-21.py |
| targeted_module_upgrader | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\trading\targeted_module_upgrader.py |
| targeted_module_upgrader_integrated_2025-06-21 | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\trading\targeted_module_upgrader_integrated_2025-06-21.py |
| debug_validation | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\debug_validation.py |
| dogbox | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\dogbox.py |
| DUPLICATE_indicator_scanner_fixed | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\DUPLICATE_indicator_scanner_fixed.py |
| equality_constrained_sqp | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\equality_constrained_sqp.py |
| execution_playbook_generator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\execution_playbook_generator.py |
| fix_alpha_decay_subscribers | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\fix_alpha_decay_subscribers.py |
| fix_eventbus_integration | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\fix_eventbus_integration.py |
| launch_phase18_production | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\launch_phase18_production.py |
| liquidity_sweep_validator | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\liquidity_sweep_validator.py |
| live_trade_feedback_injector | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\live_trade_feedback_injector.py |
| ml_model_bootstrap | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\ml_model_bootstrap.py |
| omega_system_reconstruction_fixed | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\omega_system_reconstruction_fixed.py |
| ORPHAN_test_strategy_mutation_phase13 | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\ORPHAN_test_strategy_mutation_phase13.py |
| ORPHAN_validate_phase32_execution_flow_controller | SIGNAL_PROCESSING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\ORPHAN_validate_phase32_execution_flow_controller.py |
| phase13_event_subscriber | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase13_event_subscriber.py |
| phase20_final_validation | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase20_final_validation.py |
| phase31_epl_validation | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase31_epl_validation.py |
| phase32_completion | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase32_completion.py |
| phase33_registration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase33_registration.py |
| phase34_validation | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase34_validation.py |
| phase47_live_feed_sync_integration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase47_live_feed_sync_integration.py |
| phase61_62_orchestrator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase61_62_orchestrator.py |
| phases_64_65_completion_report | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phases_64_65_completion_report.py |
| phase_50_optimization | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase_50_optimization.py |
| phase_95_completion_validator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase_95_completion_validator.py |
| phase_96_completion_validator | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase_96_completion_validator.py |
| phase_97_5_step_2_regenerate_core | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\phase_97_5_step_2_regenerate_core.py |
| post_trade_feedback_collector | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\post_trade_feedback_collector.py |
| qp_subproblem | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\qp_subproblem.py |
| quick_validation | SIGNAL_PROCESSING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\quick_validation.py |
| resolve_phase13_validation | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\resolve_phase13_validation.py |
| run_phase12_validation | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\run_phase12_validation.py |
| signal_pattern_visualizer | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\signal_pattern_visualizer.py |
| smart_execution_monitor | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\smart_execution_monitor.py |
| step7_architect_compliant_validator | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\step7_architect_compliant_validator.py |
| step7_deadlock_debugger | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\step7_deadlock_debugger.py |
| SYNTAX_event_bus_manager | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\SYNTAX_event_bus_manager.py |
| test_artist | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_artist.py |
| test_axis_nan_policy | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_axis_nan_policy.py |
| test_backtest_engine | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_backtest_engine.py |
| test_backtest_engine_phase92b | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_backtest_engine_phase92b.py |
| test_k_means | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_k_means.py |
| test_ltisys | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_ltisys.py |
| test_memmapping | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_memmapping.py |
| test_ml_retraining_loop_phase57 | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_ml_retraining_loop_phase57.py |
| test_pca | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_pca.py |
| test_phase33_execution_envelope_harmonizer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_phase33_execution_envelope_harmonizer.py |
| test_phase42_context_synthesizer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_phase42_context_synthesizer.py |
| test_phase42_macro_sync | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_phase42_macro_sync.py |
| test_phase43_sentiment_fusion_basic | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_phase43_sentiment_fusion_basic.py |
| test_phase57_58_integration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_phase57_58_integration.py |
| test_phase9_safe_reinforcement | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_phase9_safe_reinforcement.py |
| test_phase_101_sniper_integration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_phase_101_sniper_integration.py |
| test_signal_generation | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_signal_generation.py |
| test_signal_loop_reinforcement_step8 | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_signal_loop_reinforcement_step8.py |
| test_smart_execution_monitor | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_smart_execution_monitor.py |
| test_smart_monitor | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_smart_monitor.py |
| test_smart_monitor_safe | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\test_smart_monitor_safe.py |
| trade_memory_feedback_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\trade_memory_feedback_engine.py |
| trade_visualizer | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\trade_visualizer.py |
| trf | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\trf.py |
| ultra_minimal_test | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\ultra_minimal_test.py |
| validate_institutional_system | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_institutional_system.py |
| validate_phase13 | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase13.py |
| validate_phase35_completion | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase35_completion.py |
| validate_phase44_integration | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase44_integration.py |
| validate_phase47_completion | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase47_completion.py |
| validate_phase47_integration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase47_integration.py |
| validate_phase55_56_ml_control_integration | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase55_56_ml_control_integration.py |
| validate_phase57_58_complete | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase57_58_complete.py |
| validate_phase92b_completion | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase92b_completion.py |
| validate_phase_49 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase_49.py |
| validate_phase_50 | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase_50.py |
| validate_phase_82_83 | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_phase_82_83.py |
| validate_recovery | TRADING |  | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\validate_recovery.py |
| _cobyqa_py | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\_cobyqa_py.py |
| _elementwise_iterative_method | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\_elementwise_iterative_method.py |
| _emoji_codes | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\_emoji_codes.py |
| _linprog_doc | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\_linprog_doc.py |
| _linprog_highs | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\_linprog_highs.py |
| _linprog_simplex | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\_linprog_simplex.py |
| _remove_redundancy | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\_remove_redundancy.py |
| _trustregion_ncg | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\modules\unclassified\_trustregion_ncg.py |
| discovery_main | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\interface\dashboard\discovery_main.py |
| execution_console | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\interface\dashboard\components\execution_console.py |
| login_dialog | TRADING | FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\interface\dashboard\components\login_dialog.py |
| market_panel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\interface\dashboard\components\market_panel.py |
| patch_queue | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\interface\dashboard\components\patch_queue.py |
| telemetry_monitor | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\interface\dashboard\components\telemetry_monitor.py |
| mt5_connection_bridge | TRADING |  | C:\Users\patra\Genesis FINAL TRY\core\connectors\mt5_connection_bridge.py |
| advanced_dashboard_module_wiring_engine | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\advanced_dashboard_module_wiring_engine.py |
| architect_mode_system_guardian | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\architect_mode_system_guardian.py |
| boot_genesis | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\boot_genesis.py |
| complete_intelligent_module_wiring_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\complete_intelligent_module_wiring_engine.py |
| comprehensive_module_upgrade_engine | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\comprehensive_module_upgrade_engine.py |
| count_modules | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\count_modules.py |
| dashboard_engine | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\dashboard_engine.py |
| dashboard_panel_configurator | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\dashboard_panel_configurator.py |
| emergency_compliance_fixer | TRADING | MOCK_DATA_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION, MISSING_EVENTBUS_INTEGRATION | C:\Users\patra\Genesis FINAL TRY\emergency_compliance_fixer.py |
| emergency_compliance_fixer_fixed | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\emergency_compliance_fixer_fixed.py |
| emergency_module_restoration | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\emergency_module_restoration.py |
| emergency_python_rebuilder | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\emergency_python_rebuilder.py |
| eventbus_restoration | TRADING | FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\eventbus_restoration.py |
| final_compliance_auditor | UNCLASSIFIED | MISSING_EVENTBUS_INTEGRATION, MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\final_compliance_auditor.py |
| genesis_api | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_api.py |
| genesis_dashboard | DATA_PROCESSING |  | C:\Users\patra\Genesis FINAL TRY\genesis_dashboard.py |
| genesis_desktop | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_desktop.py |
| genesis_desktop_verification | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_desktop_verification.py |
| genesis_docker_dashboard | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_docker_dashboard.py |
| genesis_docker_gui_app | TRADING | MOCK_DATA_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_docker_gui_app.py |
| genesis_docker_launcher | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_docker_launcher.py |
| genesis_duplicate_verification | SIGNAL_PROCESSING | MISSING_EVENTBUS_INTEGRATION, MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\genesis_duplicate_verification.py |
| genesis_functionality_test | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_functionality_test.py |
| genesis_high_architecture_mapper | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_high_architecture_mapper.py |
| genesis_institutional_command_center_v7 | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_institutional_command_center_v7.py |
| genesis_institutional_dashboard | TRADING | MOCK_DATA_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_institutional_dashboard.py |
| genesis_minimal_launcher | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_minimal_launcher.py |
| genesis_module_upgrade_scan | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_module_upgrade_scan.py |
| genesis_optimization_engine | TRADING | MOCK_DATA_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_optimization_engine.py |
| genesis_optimization_engine_fixed | TRADING | MOCK_DATA_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_optimization_engine_fixed.py |
| genesis_production_launcher | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_production_launcher.py |
| genesis_real_mt5_integration_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_real_mt5_integration_engine.py |
| genesis_real_mt5_login_dialog | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\genesis_real_mt5_login_dialog.py |
| intelligent_module_wiring_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\intelligent_module_wiring_engine.py |
| launch | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\launch.py |
| launch_backend_simple | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\launch_backend_simple.py |
| launch_desktop_app | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION | C:\Users\patra\Genesis FINAL TRY\launch_desktop_app.py |
| launch_genesis | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\launch_genesis.py |
| launch_genesis_audit_mode | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\launch_genesis_audit_mode.py |
| module_classification_analyzer | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\module_classification_analyzer.py |
| module_integration_engine | TRADING |  | C:\Users\patra\Genesis FINAL TRY\module_integration_engine.py |
| module_scanner_panel | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\module_scanner_panel.py |
| module_tracking_report_generator | SIGNAL_PROCESSING | MOCK_DATA_VIOLATION, MISSING_EVENTBUS_INTEGRATION, MISSING_TELEMETRY | C:\Users\patra\Genesis FINAL TRY\module_tracking_report_generator.py |
| mt5_discovery_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\mt5_discovery_engine.py |
| open_genesis | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\open_genesis.py |
| orphan_module_restructure_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\orphan_module_restructure_engine.py |
| orphan_module_restructure_engine_fixed | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, STUB_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\orphan_module_restructure_engine_fixed.py |
| purge_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\purge_engine.py |
| quick_launch | TRADING |  | C:\Users\patra\Genesis FINAL TRY\quick_launch.py |
| quick_module_scanner | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\quick_module_scanner.py |
| repair_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\repair_engine.py |
| run_architecture_mapper | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\run_architecture_mapper.py |
| start-genesis-local | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\start-genesis-local.py |
| targeted_module_upgrader | TRADING | PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\targeted_module_upgrader.py |
| test_enhanced_genesis | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\test_enhanced_genesis.py |
| test_x_server | TRADING | FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\test_x_server.py |
| urgent_module_restoration_engine | TRADING | MOCK_DATA_VIOLATION, FALLBACK_LOGIC_VIOLATION, PASSIVE_CODE_VIOLATION | C:\Users\patra\Genesis FINAL TRY\urgent_module_restoration_engine.py |

## 🚀 Next Actions Required

- Review flagged modules for manual integration
- Test integrated modules functionality
- Verify EventBus connectivity
- Run compliance validation
- Update system documentation

## 🎯 Integration Recommendations

Based on the analysis, the following actions are recommended:

1. **High Priority:** Review and manually integrate flagged modules with 
   business logic
2. **Medium Priority:** Test all integrated modules for functionality
3. **Low Priority:** Clean up rejected/redundant modules

---

**Report Generated:** 2025-06-21T18:05:31.679496  
**GENESIS AI Agent v7.0.0 - Institutional-Grade Trading System**
