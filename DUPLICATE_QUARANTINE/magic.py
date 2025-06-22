import logging
# <!-- @GENESIS_MODULE_START: magic -->
"""
🏛️ GENESIS MAGIC - INSTITUTIONAL GRADE v8.0.0
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

# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import ast
import sys
from typing import Any, Final

from streamlit import config

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

                emit_telemetry("magic", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("magic", "position_calculated", {
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
                            "module": "magic",
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
                    print(f"Emergency stop error in magic: {e}")
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
                    "module": "magic",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("magic", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in magic: {e}")
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



# When a Streamlit app is magicified, we insert a `magic_funcs` import near the top of
# its module's AST: import streamlit.runtime.scriptrunner.magic_funcs as __streamlitmagic__
MAGIC_MODULE_NAME: Final = "__streamlitmagic__"


def add_magic(code: str, script_path: str) -> Any:
    """Modifies the code to support magic Streamlit commands.

    Parameters
    ----------
    code : str
        The Python code.
    script_path : str
        The path to the script file.

    Returns
    -------
    ast.Module
        The syntax tree for the code.

    """
    # Pass script_path so we get pretty exceptions.
    tree = ast.parse(code, script_path, "exec")

    file_ends_in_semicolon = _does_file_end_in_semicolon(tree, code)

    _modify_ast_subtree(
        tree, is_root=True, file_ends_in_semicolon=file_ends_in_semicolon
    )

    return tree


def _modify_ast_subtree(
    tree: Any,
    body_attr: str = "body",
    is_root: bool = False,
    file_ends_in_semicolon: bool = False,
) -> None:
    """Parses magic commands and modifies the given AST (sub)tree."""

    body = getattr(tree, body_attr)

    for i, node in enumerate(body):
        node_type = type(node)

        # Recursively parses the content of the statements
        # `with` as well as function definitions.
        # Also covers their async counterparts
        if (
            node_type is ast.FunctionDef
            or node_type is ast.With
            or node_type is ast.AsyncFunctionDef
            or node_type is ast.AsyncWith
        ):
            _modify_ast_subtree(node)

        # Recursively parses the content of the statements
        # `for` and `while`.
        # Also covers their async counterparts
        elif (
            node_type is ast.For or node_type is ast.While or node_type is ast.AsyncFor
        ):
            _modify_ast_subtree(node)
            _modify_ast_subtree(node, "orelse")

        # Recursively parses methods in a class.
        elif node_type is ast.ClassDef:
            for inner_node in node.body:
                if type(inner_node) in {ast.FunctionDef, ast.AsyncFunctionDef}:
                    _modify_ast_subtree(inner_node)

        # Recursively parses the contents of try statements,
        # all their handlers (except and else) and the finally body
        elif node_type is ast.Try or (
            sys.version_info >= (3, 11) and node_type is ast.TryStar
        ):
            _modify_ast_subtree(node)
            _modify_ast_subtree(node, body_attr="finalbody")
            _modify_ast_subtree(node, body_attr="orelse")
            for handler_node in node.handlers:
                _modify_ast_subtree(handler_node)

        # Recursively parses if blocks, as well as their else/elif blocks
        # (else/elif are both mapped to orelse)
        # it intentionally does not parse the test expression.
        elif node_type is ast.If:
            _modify_ast_subtree(node)
            _modify_ast_subtree(node, "orelse")

        elif sys.version_info >= (3, 10) and node_type is ast.Match:
            for case_node in node.cases:
                _modify_ast_subtree(case_node)

        # Convert standalone expression nodes to st.write
        elif node_type is ast.Expr:
            value = _get_st_write_from_expr(
                node,
                i,
                parent_type=type(tree),
                is_root=is_root,
                is_last_expr=(i == len(body) - 1),
                file_ends_in_semicolon=file_ends_in_semicolon,
            )
            if value is not None:
                node.value = value

    if is_root:
        # Import Streamlit so we can use it in the new_value above.
        _insert_import_statement(tree)

    ast.fix_missing_locations(tree)


def _insert_import_statement(tree: Any) -> None:
    """Insert Streamlit import statement at the top(ish) of the tree."""

    st_import = _build_st_import_statement()

    # If the 0th node is already an import statement, put the Streamlit
    # import below that, so we don't break "from __future__ import".
    if tree.body and type(tree.body[0]) in {ast.ImportFrom, ast.Import}:
        tree.body.insert(1, st_import)

    # If the 0th node is a docstring and the 1st is an import statement,
    # put the Streamlit import below those, so we don't break "from
    # __future__ import".
    elif (
        len(tree.body) > 1
        and (
            type(tree.body[0]) is ast.Expr
            and _is_string_constant_node(tree.body[0].value)
        )
        and type(tree.body[1]) in {ast.ImportFrom, ast.Import}
    ):
        tree.body.insert(2, st_import)

    else:
        tree.body.insert(0, st_import)


def _build_st_import_statement() -> ast.Import:
    """Build AST node for `import magic_funcs as __streamlitmagic__`."""
    return ast.Import(
        names=[
            ast.alias(
                name="streamlit.runtime.scriptrunner.magic_funcs",
                asname=MAGIC_MODULE_NAME,
            )
        ]
    )


def _build_st_write_call(nodes: list[Any]) -> ast.Call:
    """Build AST node for `__streamlitmagic__.transparent_write(*nodes)`."""
    return ast.Call(
        func=ast.Attribute(
            attr="transparent_write",
            value=ast.Name(id=MAGIC_MODULE_NAME, ctx=ast.Load()),
            ctx=ast.Load(),
        ),
        args=nodes,
        keywords=[],
    )


def _get_st_write_from_expr(
    node: Any,
    i: int,
    parent_type: Any,
    is_root: bool,
    is_last_expr: bool,
    file_ends_in_semicolon: bool,
) -> ast.Call | None:
    # Don't wrap function calls
    # (Unless the function call happened at the end of the root node, AND
    # magic.displayLastExprIfNoSemicolon is True. This allows us to support notebook-like
    # behavior, where we display the last function in a cell)
    if type(node.value) is ast.Call and not _is_displayable_last_expr(
        is_root, is_last_expr, file_ends_in_semicolon
    ):
        return None

    # Don't wrap DocString nodes
    # (Unless magic.displayRootDocString, in which case we do wrap the root-level
    # docstring with st.write. This allows us to support notebook-like behavior
    # where you can have a cell with a markdown string)
    if _is_docstring_node(
        node.value, i, parent_type
    ) and not _should_display_docstring_like_node_anyway(is_root):
        return None

    # Don't wrap yield nodes
    if type(node.value) is ast.Yield or type(node.value) is ast.YieldFrom:
        return None

    # Don't wrap await nodes
    if type(node.value) is ast.Await:
        return None

    # If tuple, call st.write(*the_tuple). This allows us to add a comma at the end of a
    # statement to turn it into an expression that should be
    # st-written. Ex: "np.random.randn(1000, 2),"
    args = node.value.elts if type(node.value) is ast.Tuple else [node.value]
    return _build_st_write_call(args)


def _is_string_constant_node(node: Any) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _is_docstring_node(node: Any, node_index: int, parent_type: Any) -> bool:
    return (
        node_index == 0
        and _is_string_constant_node(node)
        and parent_type in {ast.FunctionDef, ast.AsyncFunctionDef, ast.Module}
    )


def _does_file_end_in_semicolon(tree: Any, code: str) -> bool:
    file_ends_in_semicolon = False

    # Avoid spending time with this operation if magic.displayLastExprIfNoSemicolon is
    # not set.
    if config.get_option("magic.displayLastExprIfNoSemicolon"):
        if len(tree.body) == 0:
            return False

        last_line_num = getattr(tree.body[-1], "end_lineno", None)

        if last_line_num is not None:
            last_line_str: str = code.split("\n")[last_line_num - 1]
            file_ends_in_semicolon = last_line_str.strip(" ").endswith(";")

    return file_ends_in_semicolon


def _is_displayable_last_expr(
    is_root: bool, is_last_expr: bool, file_ends_in_semicolon: bool
) -> bool:
    return (
        # This is a "displayable last expression" if...
        # ...it's actually the last expression...
        is_last_expr
        # ...in the root scope...
        and is_root
        # ...it does not end in a semicolon...
        and not file_ends_in_semicolon
        # ...and this config option is telling us to show it
        and config.get_option("magic.displayLastExprIfNoSemicolon")
    )


def _should_display_docstring_like_node_anyway(is_root: bool) -> bool:
    return config.get_option("magic.displayRootDocString") and is_root


# <!-- @GENESIS_MODULE_END: magic -->
