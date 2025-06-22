import logging
# <!-- @GENESIS_MODULE_START: test_monotonic_tree -->
"""
🏛️ GENESIS TEST_MONOTONIC_TREE - INSTITUTIONAL GRADE v8.0.0
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

import numpy as np
import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (

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

                emit_telemetry("test_monotonic_tree", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_monotonic_tree", "position_calculated", {
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
                            "module": "test_monotonic_tree",
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
                    print(f"Emergency stop error in test_monotonic_tree: {e}")
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
                    "module": "test_monotonic_tree",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_monotonic_tree", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_monotonic_tree: {e}")
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


    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS

TREE_CLASSIFIER_CLASSES = [DecisionTreeClassifier, ExtraTreeClassifier]
TREE_REGRESSOR_CLASSES = [DecisionTreeRegressor, ExtraTreeRegressor]
TREE_BASED_CLASSIFIER_CLASSES = TREE_CLASSIFIER_CLASSES + [
    RandomForestClassifier,
    ExtraTreesClassifier,
]
TREE_BASED_REGRESSOR_CLASSES = TREE_REGRESSOR_CLASSES + [
    RandomForestRegressor,
    ExtraTreesRegressor,
]


@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("sparse_splitter", (True, False))
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_monotonic_constraints_classifications(
    TreeClassifier,
    depth_first_builder,
    sparse_splitter,
    global_random_seed,
    csc_container,
):
    n_samples = 1000
    n_samples_train = 900
    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        random_state=global_random_seed,
    )
    X_train, y_train = X[:n_samples_train], y[:n_samples_train]
    X_test, _ = X[n_samples_train:], y[n_samples_train:]

    X_test_0incr, X_test_0decr = np.copy(X_test), np.copy(X_test)
    X_test_1incr, X_test_1decr = np.copy(X_test), np.copy(X_test)
    X_test_0incr[:, 0] += 10
    X_test_0decr[:, 0] -= 10
    X_test_1incr[:, 1] += 10
    X_test_1decr[:, 1] -= 10
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1
    monotonic_cst[1] = -1

    if depth_first_builder:
        est = TreeClassifier(max_depth=None, monotonic_cst=monotonic_cst)
    else:
        est = TreeClassifier(
            max_depth=None,
            monotonic_cst=monotonic_cst,
            max_leaf_nodes=n_samples_train,
        )
    if hasattr(est, "random_state"):
        est.set_params(**{"random_state": global_random_seed})
    if hasattr(est, "n_estimators"):
        est.set_params(**{"n_estimators": 5})
    if sparse_splitter:
        X_train = csc_container(X_train)
    est.fit(X_train, y_train)
    proba_test = est.predict_proba(X_test)

    assert np.logical_and(proba_test >= 0.0, proba_test <= 1.0).all(), (
        "Probability should always be in [0, 1] range."
    )
    assert_allclose(proba_test.sum(axis=1), 1.0)

    # Monotonic increase constraint, it applies to the positive class
    assert np.all(est.predict_proba(X_test_0incr)[:, 1] >= proba_test[:, 1])
    assert np.all(est.predict_proba(X_test_0decr)[:, 1] <= proba_test[:, 1])

    # Monotonic decrease constraint, it applies to the positive class
    assert np.all(est.predict_proba(X_test_1incr)[:, 1] <= proba_test[:, 1])
    assert np.all(est.predict_proba(X_test_1decr)[:, 1] >= proba_test[:, 1])


@pytest.mark.parametrize("TreeRegressor", TREE_BASED_REGRESSOR_CLASSES)
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("sparse_splitter", (True, False))
@pytest.mark.parametrize("criterion", ("absolute_error", "squared_error"))
@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_monotonic_constraints_regressions(
    TreeRegressor,
    depth_first_builder,
    sparse_splitter,
    criterion,
    global_random_seed,
    csc_container,
):
    n_samples = 1000
    n_samples_train = 900
    # Build a regression task using 5 informative features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=5,
        n_informative=5,
        random_state=global_random_seed,
    )
    train = np.arange(n_samples_train)
    test = np.arange(n_samples_train, n_samples)
    X_train = X[train]
    y_train = y[train]
    X_test = np.copy(X[test])
    X_test_incr = np.copy(X_test)
    X_test_decr = np.copy(X_test)
    X_test_incr[:, 0] += 10
    X_test_decr[:, 1] += 10
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1
    monotonic_cst[1] = -1

    if depth_first_builder:
        est = TreeRegressor(
            max_depth=None,
            monotonic_cst=monotonic_cst,
            criterion=criterion,
        )
    else:
        est = TreeRegressor(
            max_depth=8,
            monotonic_cst=monotonic_cst,
            criterion=criterion,
            max_leaf_nodes=n_samples_train,
        )
    if hasattr(est, "random_state"):
        est.set_params(random_state=global_random_seed)
    if hasattr(est, "n_estimators"):
        est.set_params(**{"n_estimators": 5})
    if sparse_splitter:
        X_train = csc_container(X_train)
    est.fit(X_train, y_train)
    y = est.predict(X_test)
    # Monotonic increase constraint
    y_incr = est.predict(X_test_incr)
    # y_incr should always be greater than y
    assert np.all(y_incr >= y)

    # Monotonic decrease constraint
    y_decr = est.predict(X_test_decr)
    # y_decr should always be lower than y
    assert np.all(y_decr <= y)


@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
def test_multiclass_raises(TreeClassifier):
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=0
    )
    y[0] = 0
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = -1
    monotonic_cst[1] = 1
    est = TreeClassifier(max_depth=None, monotonic_cst=monotonic_cst, random_state=0)

    msg = "Monotonicity constraints are not supported with multiclass classification"
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)


@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
def test_multiple_output_raises(TreeClassifier):
    X = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    y = [[1, 0, 1, 0, 1], [1, 0, 1, 0, 1]]

    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-1, 1]), random_state=0
    )
    msg = "Monotonicity constraints are not supported with multiple output"
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)


@pytest.mark.parametrize(
    "Tree",
    [
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        ExtraTreeClassifier,
        ExtraTreeRegressor,
    ],
)
def test_missing_values_raises(Tree):
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, n_informative=3, random_state=0
    )
    X[0, 0] = np.nan
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1
    est = Tree(max_depth=None, monotonic_cst=monotonic_cst, random_state=0)

    msg = "Input X contains NaN"
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)


@pytest.mark.parametrize("TreeClassifier", TREE_BASED_CLASSIFIER_CLASSES)
def test_bad_monotonic_cst_raises(TreeClassifier):
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    y = [1, 0, 1, 0, 1]

    msg = "monotonic_cst has shape 3 but the input data X has 2 features."
    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-1, 1, 0]), random_state=0
    )
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)

    msg = "monotonic_cst must be None or an array-like of -1, 0 or 1."
    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-2, 2]), random_state=0
    )
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)

    est = TreeClassifier(
        max_depth=None, monotonic_cst=np.array([-1, 0.8]), random_state=0
    )
    with pytest.raises(ValueError, match=msg + "(.*)0.8]"):
        est.fit(X, y)


def assert_1d_reg_tree_children_monotonic_bounded(tree_, monotonic_sign):
    values = tree_.value
    for i in range(tree_.node_count):
        if tree_.children_left[i] > i and tree_.children_right[i] > i:
            # Check monotonicity on children
            i_left = tree_.children_left[i]
            i_right = tree_.children_right[i]
            if monotonic_sign == 1:
                assert values[i_left] <= values[i_right]
            elif monotonic_sign == -1:
                assert values[i_left] >= values[i_right]
            val_middle = (values[i_left] + values[i_right]) / 2
            # Check bounds on grand-children, filtering out leaf nodes
            if tree_.feature[i_left] >= 0:
                i_left_right = tree_.children_right[i_left]
                if monotonic_sign == 1:
                    assert values[i_left_right] <= val_middle
                elif monotonic_sign == -1:
                    assert values[i_left_right] >= val_middle
            if tree_.feature[i_right] >= 0:
                i_right_left = tree_.children_left[i_right]
                if monotonic_sign == 1:
                    assert val_middle <= values[i_right_left]
                elif monotonic_sign == -1:
                    assert val_middle >= values[i_right_left]


def test_assert_1d_reg_tree_children_monotonic_bounded():
    X = np.linspace(-1, 1, 7).reshape(-1, 1)
    y = np.sin(2 * np.pi * X.ravel())

    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)

    with pytest.raises(AssertionError):
        assert_1d_reg_tree_children_monotonic_bounded(reg.tree_, 1)

    with pytest.raises(AssertionError):
        assert_1d_reg_tree_children_monotonic_bounded(reg.tree_, -1)


def assert_1d_reg_monotonic(clf, monotonic_sign, min_x, max_x, n_steps):
    X_grid = np.linspace(min_x, max_x, n_steps).reshape(-1, 1)
    y_pred_grid = clf.predict(X_grid)
    if monotonic_sign == 1:
        assert (np.diff(y_pred_grid) >= 0.0).all()
    elif monotonic_sign == -1:
        assert (np.diff(y_pred_grid) <= 0.0).all()


@pytest.mark.parametrize("TreeRegressor", TREE_REGRESSOR_CLASSES)
def test_1d_opposite_monotonicity_cst_data(TreeRegressor):
    # Check that positive monotonic data with negative monotonic constraint
    # yield constant predictions, equal to the average of target values
    X = np.linspace(-2, 2, 10).reshape(-1, 1)
    y = X.ravel()
    clf = TreeRegressor(monotonic_cst=[-1])
    clf.fit(X, y)
    assert clf.tree_.node_count == 1
    assert clf.tree_.value[0] == 0.0

    # Swap monotonicity
    clf = TreeRegressor(monotonic_cst=[1])
    clf.fit(X, -y)
    assert clf.tree_.node_count == 1
    assert clf.tree_.value[0] == 0.0


@pytest.mark.parametrize("TreeRegressor", TREE_REGRESSOR_CLASSES)
@pytest.mark.parametrize("monotonic_sign", (-1, 1))
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("criterion", ("absolute_error", "squared_error"))
def test_1d_tree_nodes_values(
    TreeRegressor, monotonic_sign, depth_first_builder, criterion, global_random_seed
):
    # Adaptation from test_nodes_values in test_monotonic_constraints.py
    # in sklearn.ensemble._hist_gradient_boosting
    # Build a single tree with only one feature, and make sure the node
    # values respect the monotonicity constraints.

    # Considering the following tree with a monotonic +1 constraint, we
    # should have:
    #
    #       root
    #      /    \
    #     a      b
    #    / \    / \
    #   c   d  e   f
    #
    #        a <=  root  <= b
    # c <= d <= (a + b) / 2 <= e <= f

    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    n_features = 1
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    if depth_first_builder:
        # No max_leaf_nodes, default depth first tree builder
        clf = TreeRegressor(
            monotonic_cst=[monotonic_sign],
            criterion=criterion,
            random_state=global_random_seed,
        )
    else:
        # max_leaf_nodes triggers best first tree builder
        clf = TreeRegressor(
            monotonic_cst=[monotonic_sign],
            max_leaf_nodes=n_samples,
            criterion=criterion,
            random_state=global_random_seed,
        )
    clf.fit(X, y)

    assert_1d_reg_tree_children_monotonic_bounded(clf.tree_, monotonic_sign)
    assert_1d_reg_monotonic(clf, monotonic_sign, np.min(X), np.max(X), 100)


def assert_nd_reg_tree_children_monotonic_bounded(tree_, monotonic_cst):
    upper_bound = np.full(tree_.node_count, np.inf)
    lower_bound = np.full(tree_.node_count, -np.inf)
    for i in range(tree_.node_count):
        feature = tree_.feature[i]
        node_value = tree_.value[i][0][0]  # unpack value from nx1x1 array
        # While building the tree, the computed middle value is slightly
        # different from the average of the siblings values, because
        # sum_right / weighted_n_right
        # is slightly different from the value of the right sibling.
        # This can cause a discrepancy up to numerical noise when clipping,
        # which is resolved by comparing with some loss of precision.
        assert np.float32(node_value) <= np.float32(upper_bound[i])
        assert np.float32(node_value) >= np.float32(lower_bound[i])

        if feature < 0:
            # Leaf: nothing to do
            continue

        # Split node: check and update bounds for the children.
        i_left = tree_.children_left[i]
        i_right = tree_.children_right[i]
        # unpack value from nx1x1 array
        middle_value = (tree_.value[i_left][0][0] + tree_.value[i_right][0][0]) / 2

        if monotonic_cst[feature] == 0:
            # Feature without monotonicity constraint: propagate bounds
            # down the tree to both children.
            # Otherwise, with 2 features and a monotonic increase constraint
            # (encoded by +1) on feature 0, the following tree can be accepted,
            # although it does not respect the monotonic increase constraint:
            #
            #                      X[0] <= 0
            #                      value = 100
            #                     /            \
            #          X[0] <= -1                X[1] <= 0
            #          value = 50                value = 150
            #        /            \             /            \
            #    leaf           leaf           leaf          leaf
            #    value = 25     value = 75     value = 50    value = 250

            lower_bound[i_left] = lower_bound[i]
            upper_bound[i_left] = upper_bound[i]
            lower_bound[i_right] = lower_bound[i]
            upper_bound[i_right] = upper_bound[i]

        elif monotonic_cst[feature] == 1:
            # Feature with constraint: check monotonicity
            assert tree_.value[i_left] <= tree_.value[i_right]

            # Propagate bounds down the tree to both children.
            lower_bound[i_left] = lower_bound[i]
            upper_bound[i_left] = middle_value
            lower_bound[i_right] = middle_value
            upper_bound[i_right] = upper_bound[i]

        elif monotonic_cst[feature] == -1:
            # Feature with constraint: check monotonicity
            assert tree_.value[i_left] >= tree_.value[i_right]

            # Update and propagate bounds down the tree to both children.
            lower_bound[i_left] = middle_value
            upper_bound[i_left] = upper_bound[i]
            lower_bound[i_right] = lower_bound[i]
            upper_bound[i_right] = middle_value

        else:  # pragma: no cover
            raise ValueError(f"monotonic_cst[{feature}]={monotonic_cst[feature]}")


def test_assert_nd_reg_tree_children_monotonic_bounded():
    # Check that assert_nd_reg_tree_children_monotonic_bounded can detect
    # non-monotonic tree predictions.
    X = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
    y = np.sin(X).ravel()
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)

    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [1])

    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [-1])

    assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [0])

    # Check that assert_nd_reg_tree_children_monotonic_bounded raises
    # when the data (and therefore the model) is naturally monotonic in the
    # opposite direction.
    X = np.linspace(-5, 5, 5).reshape(-1, 1)
    y = X.ravel() ** 3
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, y)

    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [-1])

    # For completeness, check that the converse holds when swapping the sign.
    reg = DecisionTreeRegressor(max_depth=None, random_state=0).fit(X, -y)

    with pytest.raises(AssertionError):
        assert_nd_reg_tree_children_monotonic_bounded(reg.tree_, [1])


@pytest.mark.parametrize("TreeRegressor", TREE_REGRESSOR_CLASSES)
@pytest.mark.parametrize("monotonic_sign", (-1, 1))
@pytest.mark.parametrize("depth_first_builder", (True, False))
@pytest.mark.parametrize("criterion", ("absolute_error", "squared_error"))
def test_nd_tree_nodes_values(
    TreeRegressor, monotonic_sign, depth_first_builder, criterion, global_random_seed
):
    # Build tree with several features, and make sure the nodes
    # values respect the monotonicity constraints.

    # Considering the following tree with a monotonic increase constraint on X[0],
    # we should have:
    #
    #            root
    #           X[0]<=t
    #          /       \
    #         a         b
    #     X[0]<=u   X[1]<=v
    #    /       \   /     \
    #   c        d  e       f
    #
    # i)   a <= root <= b
    # ii)  c <= a <= d <= (a+b)/2
    # iii) (a+b)/2 <= min(e,f)
    # For iii) we check that each node value is within the proper lower and
    # upper bounds.

    rng = np.random.RandomState(global_random_seed)
    n_samples = 1000
    n_features = 2
    monotonic_cst = [monotonic_sign, 0]
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)

    if depth_first_builder:
        # No max_leaf_nodes, default depth first tree builder
        clf = TreeRegressor(
            monotonic_cst=monotonic_cst,
            criterion=criterion,
            random_state=global_random_seed,
        )
    else:
        # max_leaf_nodes triggers best first tree builder
        clf = TreeRegressor(
            monotonic_cst=monotonic_cst,
            max_leaf_nodes=n_samples,
            criterion=criterion,
            random_state=global_random_seed,
        )
    clf.fit(X, y)
    assert_nd_reg_tree_children_monotonic_bounded(clf.tree_, monotonic_cst)


# <!-- @GENESIS_MODULE_END: test_monotonic_tree -->
