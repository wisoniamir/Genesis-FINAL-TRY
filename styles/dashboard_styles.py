# @GENESIS_ORPHAN_STATUS: enhanceable
# @GENESIS_SUGGESTED_ACTION: enhance
# @GENESIS_ANALYSIS_DATE: 2025-06-20T16:45:13.478708
# @GENESIS_PROTECTION: DO_NOT_DELETE_UNTIL_REVIEWED

"""


# Initialize EventBus connection
event_bus = EventBus.get_instance()
telemetry = TelemetryManager.get_instance()

GENESIS Dashboard Styles
CSS styling for Streamlit UI components
"""

import json
import logging

from hardened_event_bus import EventBus, Event


# <!-- @GENESIS_MODULE_END: dashboard_styles -->


# <!-- @GENESIS_MODULE_START: dashboard_styles -->

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Main theme colors
COLORS = {
    "primary": "#00BFFF",   # Light blue for main accents
    "secondary": "#0080FF", # Darker blue for secondary elements
    "success": "#22A922",   # Green for positive indicators
    "warning": "#FFA500",   # Orange for warnings
    "error": "#FF4136",     # Red for errors/alerts
    "info": "#00CED1",      # Light cyan for informational elements
    "neutral": "#8A8A8A",   # Gray for neutral indicators
    "background": "#0F1218", # Dark background
    "card_bg": "#1A1E24",   # Card background
    "text": "#F5F5F5",      # Light text
    "muted_text": "#B0B0B0" # Muted text
}

# Gradients
GRADIENTS = {
    "profit": "linear-gradient(90deg, rgba(34,169,34,0.2) 0%, rgba(34,169,34,0) 100%)",
    "loss": "linear-gradient(90deg, rgba(255,65,54,0.2) 0%, rgba(255,65,54,0) 100%)",
    "neutral": "linear-gradient(90deg, rgba(138,138,138,0.1) 0%, rgba(138,138,138,0) 100%)"
}

# Base CSS styles
BASE_STYLES = """
<style>
    .main-title {
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 20px;
        color: white;
    }
    
    .subtitle {
        font-size: 20px;
        font-weight: 500;
        color: #B0B0B0;
    }
    
    .module-card {
        background-color: #1A1E24;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .module-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .module-title {
        font-size: 16px;
        font-weight: 500;
    }
    
    .status-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-active {
        background-color: rgba(34,169,34,0.2);
        color: #22A922;
    }
    
    .status-inactive {
        background-color: rgba(138,138,138,0.2);
        color: #8A8A8A;
    }
    
    .status-warning {
        background-color: rgba(255,165,0,0.2);
        color: #FFA500;
    }
    
    .status-error {
        background-color: rgba(255,65,54,0.2);
        color: #FF4136;
    }
    
    .kpi-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    .kpi-card {
        background-color: #1A1E24;
        border-radius: 8px;
        padding: 15px;
        width: 23%;
        text-align: center;
    }
    
    .kpi-value {
        font-size: 24px;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .kpi-label {
        font-size: 14px;
        color: #B0B0B0;
    }
    
    .signal-card {
        background-color: #1A1E24;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    .signal-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    
    .signal-symbol {
        font-size: 16px;
        font-weight: 600;
    }
    
    .signal-badge {
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        width: fit-content;
    }
    
    .confidence-high {
        background-color: rgba(34,169,34,0.2);
        color: #22A922;
    }
    
    .confidence-medium {
        background-color: rgba(255,165,0,0.2);
        color: #FFA500;
    }
    
    .confidence-low {
        background-color: rgba(255,65,54,0.2);
        color: #FF4136;
    }
    
    .alert-card {
        background-color: rgba(255,65,54,0.1);
        border-left: 4px solid #FF4136;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 4px;
    }
    
    .alert-title {
        font-size: 16px;
        font-weight: 500;
        color: #FF4136;
        margin-bottom: 8px;
    }
    
    .alert-message {
        font-size: 14px;
    }
    
    .tag {
        background-color: rgba(0,191,255,0.2);
        color: #00BFFF;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin-right: 5px;
        display: inline-block;
    }
    
    .kill-switch {
        background-color: #FF4136;
        color: white;
        padding: 10px 15px;
        border-radius: 4px;
        text-align: center;
        font-weight: 600;
        cursor: pointer;
        margin-bottom: 20px;
    }
      .last-update {
        font-size: 12px;
        color: #8A8A8A;
        text-align: right;
        margin-top: 10px;
    }
    
    .sidebar-header {
        font-size: 18px;
        font-weight: 600;
        margin: 10px 0;
        padding-bottom: 5px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .sidebar-section {
        font-size: 16px;
        font-weight: 500;
        margin: 15px 0 5px 0;
        color: #B0B0B0;
    }
    
    .stat-card {
        background-color: #1A1E24;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    
    .stat-title {
        font-size: 12px;
        color: #B0B0B0;
        margin-bottom: 5px;
    }
    
    .stat-value {
        font-size: 16px;
        font-weight: 600;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    
    .metric-label {
        color: #B0B0B0;
    }
    
    .metric-value {
        font-weight: 500;
    }
    
    .progress-bar-container {
        height: 6px;
        background-color: rgba(138,138,138,0.2);
        border-radius: 3px;
        margin-bottom: 15px;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 3px;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Override Streamlit base styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #1A1E24;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00BFFF;
        color: white;
    }
    
    div[data-testid="stVerticalBlock"] div:has(div.stButton > button:first-child) {
        display: flex;
        justify-content: center;
    }
</style>
"""

# CSS for dark mode
DARK_MODE = """
<style>
    .stApp {
        background-color: #0F1218;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #1A1E24;
        border-radius: 0px 0px 4px 4px;
        padding: 15px;
    }
</style>
"""

# CSS for light mode
LIGHT_MODE = """
<style>
    .stApp {
        background-color: #F0F2F6;
    }
    
    .module-card, .kpi-card, .signal-card {
        background-color: white;
    }
    
    .main-title {
        color: #0F1218;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        border-radius: 0px 0px 4px 4px;
        padding: 15px;
    }
</style>
"""

def apply_styles(st, theme="dark"):
    """Apply styling to the Streamlit app"""
    st.markdown(BASE_STYLES, unsafe_allow_html=True)
    if theme == "dark":
        st.markdown(DARK_MODE, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_MODE, unsafe_allow_html=True)


def apply_dashboard_theme(theme="dark"):
    """Apply theme to dashboard - wrapper for apply_styles for compatibility"""
    try:
        # Save theme selection to session state
        if hasattr(st, 'session_state'):
            st.session_state.theme = theme
        
        # Apply theme CSS
        if theme == "light":
            st.markdown(LIGHT_MODE, unsafe_allow_html=True)
        else:
            st.markdown(DARK_MODE, unsafe_allow_html=True)
            
        logger.info(f"Applied {theme} theme to dashboard")
        
        # Send telemetry about theme change
        send_theme_telemetry(theme)
    except Exception as e:
        logger.error(f"Error applying theme: {str(e)}")


def send_theme_telemetry(theme):
    """Send telemetry about theme change"""
    try:
        # In a full implementation this would use the event bus
        logger.info(f"Theme changed to {theme}")
    except Exception as e:
        logger.error(f"Error sending theme telemetry: {str(e)}")
    if theme == "dark":
        st.markdown(DARK_MODE, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_MODE, unsafe_allow_html=True)

def module_status_badge(status):
    """Return HTML for module status badge"""
    if status == "active":
        return f'<span class="status-badge status-active">Active</span>'
    elif status == "inactive":
        return f'<span class="status-badge status-inactive">Inactive</span>'
    elif status == "warning":
        return f'<span class="status-badge status-warning">Warning</span>'
    elif status == "error":
        return f'<span class="status-badge status-error">Error</span>'
    else:
        return f'<span class="status-badge status-inactive">Unknown</span>'

def confidence_badge(score):
    """Return HTML for confidence score badge"""
    if score >= 80:
        return f'<span class="signal-badge confidence-high">{score}% Confidence</span>'
    elif score >= 60:
        return f'<span class="signal-badge confidence-medium">{score}% Confidence</span>'
    else:
        return f'<span class="signal-badge confidence-low">{score}% Confidence</span>'

def tag_badge(tag_name):
    """Return HTML for tag badge"""
    return f'<span class="tag">{tag_name}</span>'

def create_alert_card(title, message):
    """Return HTML for alert card"""
    return f'''
    <div class="alert-card">
        <div class="alert-title">{title}</div>
        <div class="alert-message">{message}</div>
    </div>
    '''



def emit_event(event_type: str, data: dict) -> None:
    """Emit event to the EventBus"""
    event = Event(event_type=event_type, source=__name__, data=data)
    event_bus.emit(event)
    telemetry.log_event(TelemetryEvent(category="module_event", name=event_type, properties=data))


def detect_divergence(price_data: list, indicator_data: list, window: int = 10) -> Dict:
    """
    Detect regular and hidden divergences between price and indicator
    
    Args:
        price_data: List of price values (closing prices)
        indicator_data: List of indicator values (e.g., RSI, MACD)
        window: Number of periods to check for divergence
        
    Returns:
        Dictionary with divergence information
    """
    result = {
        "regular_bullish": False,
        "regular_bearish": False,
        "hidden_bullish": False,
        "hidden_bearish": False,
        "strength": 0.0
    }
    
    # Need at least window + 1 periods of data
    if len(price_data) < window + 1 or len(indicator_data) < window + 1:
        return result
        
    # Get the current and historical points
    current_price = price_data[-1]
    previous_price = min(price_data[-window:-1]) if price_data[-1] > price_data[-2] else max(price_data[-window:-1])
    previous_price_idx = price_data[-window:-1].index(previous_price) + len(price_data) - window
    
    current_indicator = indicator_data[-1]
    previous_indicator = indicator_data[previous_price_idx]
    
    # Check for regular divergences
    # Bullish - Lower price lows but higher indicator lows
    if current_price < previous_price and current_indicator > previous_indicator:
        result["regular_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Higher price highs but lower indicator highs
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["regular_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Check for hidden divergences
    # Bullish - Higher price lows but lower indicator lows
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["hidden_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Lower price highs but higher indicator highs
    elif current_price < previous_price and current_indicator > previous_indicator:
        result["hidden_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Emit divergence event if detected
    if any([result["regular_bullish"], result["regular_bearish"], 
            result["hidden_bullish"], result["hidden_bearish"]]):
        emit_event("divergence_detected", {
            "type": next(k for k, v in result.items() if v is True and k != "strength"),
            "strength": result["strength"],
            "symbol": price_data.symbol if hasattr(price_data, "symbol") else "unknown",
            "timestamp": datetime.now().isoformat()
        })
        
    return result


def setup_event_subscriptions(self):
    """Set up EventBus subscriptions for this UI component"""
    event_bus.subscribe("market_data_updated", self.handle_market_data_update)
    event_bus.subscribe("trade_executed", self.handle_trade_update)
    event_bus.subscribe("position_changed", self.handle_position_update)
    event_bus.subscribe("risk_threshold_warning", self.handle_risk_warning)
    event_bus.subscribe("system_status_changed", self.handle_system_status_update)
    
    # Register with telemetry
    telemetry.log_event(TelemetryEvent(
        category="ui", 
        name="event_subscriptions_setup", 
        properties={"component": self.__class__.__name__}
    ))
