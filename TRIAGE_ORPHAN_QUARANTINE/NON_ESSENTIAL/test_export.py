import logging
# <!-- @GENESIS_MODULE_START: test_export -->
"""
🏛️ GENESIS TEST_EXPORT - INSTITUTIONAL GRADE v8.0.0
===============================================================
ARCHITECT MODE ULTIMATE: Enhanced via Complete Intelligent Wiring Engine

🎯 ENHANCED FEATURES:
- Complete EventBus integration
- Real-time telemetry monitoring
- FTMO compliance enforcement
- Emergency kill-switch protection
- Institutional-grade architecture

🔐 ARCHITECT MODE v8.0.0: Ultimate compliance enforcement
"""


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

                emit_telemetry("test_export", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_export", "position_calculated", {
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
                            "module": "test_export",
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
                    print(f"Emergency stop error in test_export: {e}")
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
                    "module": "test_export",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_export", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_export: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


from datetime import datetime


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


"""
Testing for export functions of decision trees (sklearn.tree.export).
"""

from io import StringIO
from re import finditer, search
from textwrap import dedent

import numpy as np
import pytest
from numpy.random import RandomState

from sklearn.base import is_classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_graphviz,
    export_text,
    plot_tree,
)

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
y2 = [[-1, 1], [-1, 1], [-1, 1], [1, 2], [1, 2], [1, 3]]
w = [1, 1, 1, 0.5, 0.5, 0.5]
y_degraded = [1, 1, 1, 1, 1, 1]


def test_graphviz_toy():
    # Check correctness of export_graphviz
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="gini", random_state=2
    )
    clf.fit(X, y)

    # Test export code
    contents1 = export_graphviz(clf, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )
    assert contents1 == contents2

    # Test with feature_names
    contents1 = export_graphviz(
        clf, feature_names=["feature0", "feature1"], out_file=None
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="feature0 <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test with feature_names (escaped)
    contents1 = export_graphviz(
        clf, feature_names=['feature"0"', 'feature"1"'], out_file=None
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="feature\\"0\\" <= 0.0\\n'
        "gini = 0.5\\nsamples = 6\\n"
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test with class_names
    contents1 = export_graphviz(clf, class_names=["yes", "no"], out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = yes"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]\\n'
        'class = yes"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]\\n'
        'class = no"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test with class_names (escaped)
    contents1 = export_graphviz(clf, class_names=['"yes"', '"no"'], out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = \\"yes\\""] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]\\n'
        'class = \\"yes\\""] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]\\n'
        'class = \\"no\\""] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test plot_options
    contents1 = export_graphviz(
        clf,
        filled=True,
        impurity=False,
        proportion=True,
        special_characters=True,
        rounded=True,
        out_file=None,
        fontname="sans",
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled, rounded", color="black", '
        'fontname="sans"] ;\n'
        'edge [fontname="sans"] ;\n'
        "0 [label=<x<SUB>0</SUB> &le; 0.0<br/>samples = 100.0%<br/>"
        'value = [0.5, 0.5]>, fillcolor="#ffffff"] ;\n'
        "1 [label=<samples = 50.0%<br/>value = [1.0, 0.0]>, "
        'fillcolor="#e58139"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        "2 [label=<samples = 50.0%<br/>value = [0.0, 1.0]>, "
        'fillcolor="#399de5"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test max_depth
    contents1 = export_graphviz(clf, max_depth=0, class_names=True, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = y[0]"] ;\n'
        '1 [label="(...)"] ;\n'
        "0 -> 1 ;\n"
        '2 [label="(...)"] ;\n'
        "0 -> 2 ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test max_depth with plot_options
    contents1 = export_graphviz(
        clf, max_depth=0, filled=True, out_file=None, node_ids=True
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="node #0\\nx[0] <= 0.0\\ngini = 0.5\\n'
        'samples = 6\\nvalue = [3, 3]", fillcolor="#ffffff"] ;\n'
        '1 [label="(...)", fillcolor="#C0C0C0"] ;\n'
        "0 -> 1 ;\n"
        '2 [label="(...)", fillcolor="#C0C0C0"] ;\n'
        "0 -> 2 ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test multi-output with weighted samples
    clf = DecisionTreeClassifier(
        max_depth=2, min_samples_split=2, criterion="gini", random_state=2
    )
    clf = clf.fit(X, y2, sample_weight=w)

    contents1 = export_graphviz(clf, filled=True, impurity=False, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\nsamples = 6\\n'
        "value = [[3.0, 1.5, 0.0]\\n"
        '[3.0, 1.0, 0.5]]", fillcolor="#ffffff"] ;\n'
        '1 [label="samples = 3\\nvalue = [[3, 0, 0]\\n'
        '[3, 0, 0]]", fillcolor="#e58139"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="x[0] <= 1.5\\nsamples = 3\\n'
        "value = [[0.0, 1.5, 0.0]\\n"
        '[0.0, 1.0, 0.5]]", fillcolor="#f1bd97"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        '3 [label="samples = 2\\nvalue = [[0, 1, 0]\\n'
        '[0, 1, 0]]", fillcolor="#e58139"] ;\n'
        "2 -> 3 ;\n"
        '4 [label="samples = 1\\nvalue = [[0.0, 0.5, 0.0]\\n'
        '[0.0, 0.0, 0.5]]", fillcolor="#e58139"] ;\n'
        "2 -> 4 ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test regression output with plot_options
    clf = DecisionTreeRegressor(
        max_depth=3, min_samples_split=2, criterion="squared_error", random_state=2
    )
    clf.fit(X, y)

    contents1 = export_graphviz(
        clf,
        filled=True,
        leaves_parallel=True,
        out_file=None,
        rotate=True,
        rounded=True,
        fontname="sans",
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled, rounded", color="black", '
        'fontname="sans"] ;\n'
        "graph [ranksep=equally, splines=polyline] ;\n"
        'edge [fontname="sans"] ;\n'
        "rankdir=LR ;\n"
        '0 [label="x[0] <= 0.0\\nsquared_error = 1.0\\nsamples = 6\\n'
        'value = 0.0", fillcolor="#f2c09c"] ;\n'
        '1 [label="squared_error = 0.0\\nsamples = 3\\'
        'nvalue = -1.0", '
        'fillcolor="#ffffff"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=-45, "
        'headlabel="True"] ;\n'
        '2 [label="squared_error = 0.0\\nsamples = 3\\nvalue = 1.0", '
        'fillcolor="#e58139"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=45, "
        'headlabel="False"] ;\n'
        "{rank=same ; 0} ;\n"
        "{rank=same ; 1; 2} ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test classifier with degraded learning set
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y_degraded)

    contents1 = export_graphviz(clf, filled=True, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="gini = 0.0\\nsamples = 6\\nvalue = 6.0", '
        'fillcolor="#ffffff"] ;\n'
        "}"
    )


@pytest.mark.parametrize("constructor", [list, np.array])
def test_graphviz_feature_class_names_array_support(constructor):
    # Check that export_graphviz treats feature names
    # and class names correctly and supports arrays
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="gini", random_state=2
    )
    clf.fit(X, y)

    # Test with feature_names
    contents1 = export_graphviz(
        clf, feature_names=constructor(["feature0", "feature1"]), out_file=None
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="feature0 <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test with class_names
    contents1 = export_graphviz(
        clf, class_names=constructor(["yes", "no"]), out_file=None
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = yes"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]\\n'
        'class = yes"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]\\n'
        'class = no"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2


def test_graphviz_errors():
    # Check for errors of export_graphviz
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2)

    # Check not-fitted decision tree error
    out = StringIO()
    with pytest.raises(NotFittedError):
        export_graphviz(clf, out)

    clf.fit(X, y)

    # Check if it errors when length of feature_names
    # mismatches with number of features
    message = "Length of feature_names, 1 does not match number of features, 2"
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=["a"])

    message = "Length of feature_names, 3 does not match number of features, 2"
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=["a", "b", "c"])

    # Check error when argument is not an estimator
    message = "is not an estimator instance"
    with pytest.raises(TypeError, match=message):
        export_graphviz(clf.fit(X, y).tree_)

    # Check class_names error
    out = StringIO()
    with pytest.raises(IndexError):
        export_graphviz(clf, out, class_names=[])


def test_friedman_mse_in_graphviz():
    clf = DecisionTreeRegressor(criterion="friedman_mse", random_state=0)
    clf.fit(X, y)
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data)

    clf = GradientBoostingClassifier(n_estimators=2, random_state=0)
    clf.fit(X, y)
    for estimator in clf.estimators_:
        export_graphviz(estimator[0], out_file=dot_data)

    for finding in finditer(r"\[.*?samples.*?\]", dot_data.getvalue()):
        assert "friedman_mse" in finding.group()


def test_precision():
    rng_reg = RandomState(2)
    rng_clf = RandomState(8)
    for X, y, clf in zip(
        (rng_reg.random_sample((5, 2)), rng_clf.random_sample((1000, 4))),
        (rng_reg.random_sample((5,)), rng_clf.randint(2, size=(1000,))),
        (
            DecisionTreeRegressor(
                criterion="friedman_mse", random_state=0, max_depth=1
            ),
            DecisionTreeClassifier(max_depth=1, random_state=0),
        ),
    ):
        clf.fit(X, y)
        for precision in (4, 3):
            dot_data = export_graphviz(
                clf, out_file=None, precision=precision, proportion=True
            )

            # With the current random state, the impurity and the threshold
            # will have the number of precision set in the export_graphviz
            # function. We will check the number of precision with a strict
            # equality. The value reported will have only 2 precision and
            # therefore, only a less equal comparison will be done.

            # check value
            for finding in finditer(r"value = \d+\.\d+", dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) <= precision + 1
            # check impurity
            if is_classifier(clf):
                pattern = r"gini = \d+\.\d+"
            else:
                pattern = r"friedman_mse = \d+\.\d+"

            # check impurity
            for finding in finditer(pattern, dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) == precision + 1
            # check threshold
            for finding in finditer(r"<= \d+\.\d+", dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) == precision + 1


def test_export_text_errors():
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    err_msg = "feature_names must contain 2 elements, got 1"
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, feature_names=["a"])
    err_msg = (
        "When `class_names` is an array, it should contain as"
        " many items as `decision_tree.classes_`. Got 1 while"
        " the tree was fitted with 2 classes."
    )
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, class_names=["a"])


def test_export_text():
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: -1
    |--- feature_1 >  0.00
    |   |--- class: 1
    """
    ).lstrip()

    assert export_text(clf) == expected_report
    # testing that leaves at level 1 are not truncated
    assert export_text(clf, max_depth=0) == expected_report
    # testing that the rest of the tree is truncated
    assert export_text(clf, max_depth=10) == expected_report

    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- weights: [3.00, 0.00] class: -1
    |--- feature_1 >  0.00
    |   |--- weights: [0.00, 3.00] class: 1
    """
    ).lstrip()
    assert export_text(clf, show_weights=True) == expected_report

    expected_report = dedent(
        """
    |- feature_1 <= 0.00
    | |- class: -1
    |- feature_1 >  0.00
    | |- class: 1
    """
    ).lstrip()
    assert export_text(clf, spacing=1) == expected_report

    X_l = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-1, 1]]
    y_l = [-1, -1, -1, 1, 1, 1, 2]
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    clf.fit(X_l, y_l)
    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: -1
    |--- feature_1 >  0.00
    |   |--- truncated branch of depth 2
    """
    ).lstrip()
    assert export_text(clf, max_depth=0) == expected_report

    X_mo = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y_mo = [[-1, -1], [-1, -1], [-1, -1], [1, 1], [1, 1], [1, 1]]

    reg = DecisionTreeRegressor(max_depth=2, random_state=0)
    reg.fit(X_mo, y_mo)

    expected_report = dedent(
        """
    |--- feature_1 <= 0.0
    |   |--- value: [-1.0, -1.0]
    |--- feature_1 >  0.0
    |   |--- value: [1.0, 1.0]
    """
    ).lstrip()
    assert export_text(reg, decimals=1) == expected_report
    assert export_text(reg, decimals=1, show_weights=True) == expected_report

    X_single = [[-2], [-1], [-1], [1], [1], [2]]
    reg = DecisionTreeRegressor(max_depth=2, random_state=0)
    reg.fit(X_single, y_mo)

    expected_report = dedent(
        """
    |--- first <= 0.0
    |   |--- value: [-1.0, -1.0]
    |--- first >  0.0
    |   |--- value: [1.0, 1.0]
    """
    ).lstrip()
    assert export_text(reg, decimals=1, feature_names=["first"]) == expected_report
    assert (
        export_text(reg, decimals=1, show_weights=True, feature_names=["first"])
        == expected_report
    )


@pytest.mark.parametrize("constructor", [list, np.array])
def test_export_text_feature_class_names_array_support(constructor):
    # Check that export_graphviz treats feature names
    # and class names correctly and supports arrays
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    expected_report = dedent(
        """
    |--- b <= 0.00
    |   |--- class: -1
    |--- b >  0.00
    |   |--- class: 1
    """
    ).lstrip()
    assert export_text(clf, feature_names=constructor(["a", "b"])) == expected_report

    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: cat
    |--- feature_1 >  0.00
    |   |--- class: dog
    """
    ).lstrip()
    assert export_text(clf, class_names=constructor(["cat", "dog"])) == expected_report


def test_plot_tree_entropy(pyplot):
    # mostly smoke tests
    # Check correctness of export_graphviz for criterion = entropy
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="entropy", random_state=2
    )
    clf.fit(X, y)

    # Test export code
    feature_names = ["first feat", "sepal_width"]
    nodes = plot_tree(clf, feature_names=feature_names)
    assert len(nodes) == 5
    assert (
        nodes[0].get_text()
        == "first feat <= 0.0\nentropy = 1.0\nsamples = 6\nvalue = [3, 3]"
    )
    assert nodes[1].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [3, 0]"
    assert nodes[2].get_text() == "True  "
    assert nodes[3].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [0, 3]"
    assert nodes[4].get_text() == "  False"


@pytest.mark.parametrize("fontsize", [None, 10, 20])
def test_plot_tree_gini(pyplot, fontsize):
    # mostly smoke tests
    # Check correctness of export_graphviz for criterion = gini
    clf = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=2,
        criterion="gini",
        random_state=2,
    )
    clf.fit(X, y)

    # Test export code
    feature_names = ["first feat", "sepal_width"]
    nodes = plot_tree(clf, feature_names=feature_names, fontsize=fontsize)
    assert len(nodes) == 5
    if fontsize is not None:
        assert all(node.get_fontsize() == fontsize for node in nodes)
    assert (
        nodes[0].get_text()
        == "first feat <= 0.0\ngini = 0.5\nsamples = 6\nvalue = [3, 3]"
    )
    assert nodes[1].get_text() == "gini = 0.0\nsamples = 3\nvalue = [3, 0]"
    assert nodes[2].get_text() == "True  "
    assert nodes[3].get_text() == "gini = 0.0\nsamples = 3\nvalue = [0, 3]"
    assert nodes[4].get_text() == "  False"


def test_not_fitted_tree(pyplot):
    # Testing if not fitted tree throws the correct error
    clf = DecisionTreeRegressor()
    with pytest.raises(NotFittedError):
        plot_tree(clf)


# <!-- @GENESIS_MODULE_END: test_export -->
