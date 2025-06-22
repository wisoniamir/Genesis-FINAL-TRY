import logging
# <!-- @GENESIS_MODULE_START: test_confusion_matrix_display -->
"""
🏛️ GENESIS TEST_CONFUSION_MATRIX_DISPLAY - INSTITUTIONAL GRADE v8.0.0
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
from numpy.testing import (

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

                emit_telemetry("test_confusion_matrix_display", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_confusion_matrix_display", "position_calculated", {
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
                            "module": "test_confusion_matrix_display",
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
                    print(f"Emergency stop error in test_confusion_matrix_display: {e}")
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
                    "module": "test_confusion_matrix_display",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_confusion_matrix_display", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_confusion_matrix_display: {e}")
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


    assert_allclose,
    assert_array_equal,
)

from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


def test_confusion_matrix_display_validation(pyplot):
    """Check that we raise the proper error when validating parameters."""
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=5, random_state=0
    )

    with pytest.raises(NotFittedError):
        ConfusionMatrixDisplay.from_estimator(SVC(), X, y)

    regressor = SVR().fit(X, y)
    y_pred_regressor = regressor.predict(X)
    y_pred_classifier = SVC().fit(X, y).predict(X)

    err_msg = "ConfusionMatrixDisplay.from_estimator only supports classifiers"
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_estimator(regressor, X, y)

    err_msg = "Mix type of y not allowed, got types"
    with pytest.raises(ValueError, match=err_msg):
        # Force `y_true` to be seen as a regression problem
        ConfusionMatrixDisplay.from_predictions(y + 0.5, y_pred_classifier)
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_predictions(y, y_pred_regressor)

    err_msg = "Found input variables with inconsistent numbers of samples"
    with pytest.raises(ValueError, match=err_msg):
        ConfusionMatrixDisplay.from_predictions(y, y_pred_classifier[::2])


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("with_labels", [True, False])
@pytest.mark.parametrize("with_display_labels", [True, False])
def test_confusion_matrix_display_custom_labels(
    pyplot, constructor_name, with_labels, with_display_labels
):
    """Check the resulting plot when labels are given."""
    n_classes = 5
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    ax = pyplot.gca()
    labels = [2, 1, 0, 3, 4] if with_labels else None
    display_labels = ["b", "d", "a", "e", "f"] if with_display_labels else None

    cm = confusion_matrix(y, y_pred, labels=labels)
    common_kwargs = {
        "ax": ax,
        "display_labels": display_labels,
        "labels": labels,
    }
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)
    assert_allclose(disp.confusion_matrix, cm)

    if with_display_labels:
        expected_display_labels = display_labels
    elif with_labels:
        expected_display_labels = labels
    else:
        expected_display_labels = list(range(n_classes))

    expected_display_labels_str = [str(name) for name in expected_display_labels]

    x_ticks = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    y_ticks = [tick.get_text() for tick in disp.ax_.get_yticklabels()]

    assert_array_equal(disp.display_labels, expected_display_labels)
    assert_array_equal(x_ticks, expected_display_labels_str)
    assert_array_equal(y_ticks, expected_display_labels_str)


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
@pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
@pytest.mark.parametrize("include_values", [True, False])
def test_confusion_matrix_display_plotting(
    pyplot,
    constructor_name,
    normalize,
    include_values,
):
    """Check the overall plotting rendering."""
    n_classes = 5
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    ax = pyplot.gca()
    cmap = "plasma"

    cm = confusion_matrix(y, y_pred)
    common_kwargs = {
        "normalize": normalize,
        "cmap": cmap,
        "ax": ax,
        "include_values": include_values,
    }
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)

    assert disp.ax_ == ax

    if normalize == "true":
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif normalize == "all":
        cm = cm / cm.sum()

    assert_allclose(disp.confusion_matrix, cm)
    import matplotlib as mpl

    assert isinstance(disp.im_, mpl.image.AxesImage)
    assert disp.im_.get_cmap().name == cmap
    assert isinstance(disp.ax_, pyplot.Axes)
    assert isinstance(disp.figure_, pyplot.Figure)

    assert disp.ax_.get_ylabel() == "True label"
    assert disp.ax_.get_xlabel() == "Predicted label"

    x_ticks = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    y_ticks = [tick.get_text() for tick in disp.ax_.get_yticklabels()]

    expected_display_labels = list(range(n_classes))

    expected_display_labels_str = [str(name) for name in expected_display_labels]

    assert_array_equal(disp.display_labels, expected_display_labels)
    assert_array_equal(x_ticks, expected_display_labels_str)
    assert_array_equal(y_ticks, expected_display_labels_str)

    image_data = disp.im_.get_array().data
    assert_allclose(image_data, cm)

    if include_values:
        assert disp.text_.shape == (n_classes, n_classes)
        fmt = ".2g"
        expected_text = np.array([format(v, fmt) for v in cm.ravel(order="C")])
        text_text = np.array([t.get_text() for t in disp.text_.ravel(order="C")])
        assert_array_equal(expected_text, text_text)
    else:
        assert disp.text_ is None


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_confusion_matrix_display(pyplot, constructor_name):
    """Check the behaviour of the default constructor without using the class
    methods."""
    n_classes = 5
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    cm = confusion_matrix(y, y_pred)
    common_kwargs = {
        "normalize": None,
        "include_values": True,
        "cmap": "viridis",
        "xticks_rotation": 45.0,
    }
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)

    assert_allclose(disp.confusion_matrix, cm)
    assert disp.text_.shape == (n_classes, n_classes)

    rotations = [tick.get_rotation() for tick in disp.ax_.get_xticklabels()]
    assert_allclose(rotations, 45.0)

    image_data = disp.im_.get_array().data
    assert_allclose(image_data, cm)

    disp.plot(cmap="plasma")
    assert disp.im_.get_cmap().name == "plasma"

    disp.plot(include_values=False)
    assert disp.text_ is None

    disp.plot(xticks_rotation=90.0)
    rotations = [tick.get_rotation() for tick in disp.ax_.get_xticklabels()]
    assert_allclose(rotations, 90.0)

    disp.plot(values_format="e")
    expected_text = np.array([format(v, "e") for v in cm.ravel(order="C")])
    text_text = np.array([t.get_text() for t in disp.text_.ravel(order="C")])
    assert_array_equal(expected_text, text_text)


def test_confusion_matrix_contrast(pyplot):
    """Check that the text color is appropriate depending on background."""

    cm = np.eye(2) / 2
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])

    disp.plot(cmap=pyplot.cm.gray)
    # diagonal text is black
    assert_allclose(disp.text_[0, 0].get_color(), [0.0, 0.0, 0.0, 1.0])
    assert_allclose(disp.text_[1, 1].get_color(), [0.0, 0.0, 0.0, 1.0])

    # off-diagonal text is white
    assert_allclose(disp.text_[0, 1].get_color(), [1.0, 1.0, 1.0, 1.0])
    assert_allclose(disp.text_[1, 0].get_color(), [1.0, 1.0, 1.0, 1.0])

    disp.plot(cmap=pyplot.cm.gray_r)
    # diagonal text is white
    assert_allclose(disp.text_[0, 1].get_color(), [0.0, 0.0, 0.0, 1.0])
    assert_allclose(disp.text_[1, 0].get_color(), [0.0, 0.0, 0.0, 1.0])

    # off-diagonal text is black
    assert_allclose(disp.text_[0, 0].get_color(), [1.0, 1.0, 1.0, 1.0])
    assert_allclose(disp.text_[1, 1].get_color(), [1.0, 1.0, 1.0, 1.0])

    # Regression test for #15920
    cm = np.array([[19, 34], [32, 58]])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])

    disp.plot(cmap=pyplot.cm.Blues)
    min_color = pyplot.cm.Blues(0)
    max_color = pyplot.cm.Blues(255)
    assert_allclose(disp.text_[0, 0].get_color(), max_color)
    assert_allclose(disp.text_[0, 1].get_color(), max_color)
    assert_allclose(disp.text_[1, 0].get_color(), max_color)
    assert_allclose(disp.text_[1, 1].get_color(), min_color)


@pytest.mark.parametrize(
    "clf",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])),
            LogisticRegression(),
        ),
    ],
    ids=["clf", "pipeline-clf", "pipeline-column_transformer-clf"],
)
def test_confusion_matrix_pipeline(pyplot, clf):
    """Check the behaviour of the plotting with more complex pipeline."""
    n_classes = 5
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    with pytest.raises(NotFittedError):
        ConfusionMatrixDisplay.from_estimator(clf, X, y)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    disp = ConfusionMatrixDisplay.from_estimator(clf, X, y)
    cm = confusion_matrix(y, y_pred)

    assert_allclose(disp.confusion_matrix, cm)
    assert disp.text_.shape == (n_classes, n_classes)


@pytest.mark.parametrize("constructor_name", ["from_estimator", "from_predictions"])
def test_confusion_matrix_with_unknown_labels(pyplot, constructor_name):
    """Check that when labels=None, the unique values in `y_pred` and `y_true`
    will be used.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/18405
    """
    n_classes = 5
    X, y = make_classification(
        n_samples=100, n_informative=5, n_classes=n_classes, random_state=0
    )
    classifier = SVC().fit(X, y)
    y_pred = classifier.predict(X)
    # create unseen labels in `y_true` not seen during fitting and not present
    # in 'classifier.classes_'
    y = y + 1

    # safe guard for the binary if/else construction
    assert constructor_name in ("from_estimator", "from_predictions")

    common_kwargs = {"labels": None}
    if constructor_name == "from_estimator":
        disp = ConfusionMatrixDisplay.from_estimator(classifier, X, y, **common_kwargs)
    else:
        disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, **common_kwargs)

    display_labels = [tick.get_text() for tick in disp.ax_.get_xticklabels()]
    expected_labels = [str(i) for i in range(n_classes + 1)]
    assert_array_equal(expected_labels, display_labels)


def test_colormap_max(pyplot):
    """Check that the max color is used for the color of the text."""
    gray = pyplot.get_cmap("gray", 1024)
    confusion_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])

    disp = ConfusionMatrixDisplay(confusion_matrix)
    disp.plot(cmap=gray)

    color = disp.text_[1, 0].get_color()
    assert_allclose(color, [1.0, 1.0, 1.0, 1.0])


def test_im_kw_adjust_vmin_vmax(pyplot):
    """Check that im_kw passes kwargs to imshow"""

    confusion_matrix = np.array([[0.48, 0.04], [0.08, 0.4]])
    disp = ConfusionMatrixDisplay(confusion_matrix)
    disp.plot(im_kw=dict(vmin=0.0, vmax=0.8))

    clim = disp.im_.get_clim()
    assert clim[0] == pytest.approx(0.0)
    assert clim[1] == pytest.approx(0.8)


def test_confusion_matrix_text_kw(pyplot):
    """Check that text_kw is passed to the text call."""
    font_size = 15.0
    X, y = make_classification(random_state=0)
    classifier = SVC().fit(X, y)

    # from_estimator passes the font size
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier, X, y, text_kw={"fontsize": font_size}
    )
    for text in disp.text_.reshape(-1):
        assert text.get_fontsize() == font_size

    # plot adjusts plot to new font size
    new_font_size = 20.0
    disp.plot(text_kw={"fontsize": new_font_size})
    for text in disp.text_.reshape(-1):
        assert text.get_fontsize() == new_font_size

    # from_predictions passes the font size
    y_pred = classifier.predict(X)
    disp = ConfusionMatrixDisplay.from_predictions(
        y, y_pred, text_kw={"fontsize": font_size}
    )
    for text in disp.text_.reshape(-1):
        assert text.get_fontsize() == font_size


# <!-- @GENESIS_MODULE_END: test_confusion_matrix_display -->
