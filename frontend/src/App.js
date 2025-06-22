/**
 * 🚀 GENESIS Dashboard - Main Application Component
 * Architect Mode v7.0 - Real-time Trading Interface
 */

import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Paper,
  Box,
  Tab,
  Tabs,
  Card,
  CardContent,
  Chip,
  Button,
  Alert
} from '@mui/material';
import {
  Timeline,
  Speed,
  Security,
  TrendingUp,
  AccountBalance,
  Settings,
  Warning
} from '@mui/icons-material';
import io from 'socket.io-client';
import SystemPanel from './components/SystemPanel';
import TelemetryPanel from './components/TelemetryPanel';
import SignalPanel from './components/SignalPanel';
import TradingPanel from './components/TradingPanel';
import RiskPanel from './components/RiskPanel';
import './App.css';

// Dark theme configuration
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff88',
    },
    secondary: {
      main: '#ff6b6b',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
  },
  typography: {
    fontFamily: '"Roboto Mono", monospace',
    h4: {
      fontWeight: 600,
      color: '#00ff88',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: '12px',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: '12px',
        },
      },
    },
  },
});

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [currentTab, setCurrentTab] = useState(0);
  const [socket, setSocket] = useState(null);
  const [systemStatus, setSystemStatus] = useState('disconnected');
  const [telemetryData, setTelemetryData] = useState({});
  const [signals, setSignals] = useState([]);
  const [orders, setOrders] = useState([]);
  const [killSwitchActive, setKillSwitchActive] = useState(false);

  useEffect(() => {
    // Initialize WebSocket connection
    const newSocket = io('http://localhost:8000');

    newSocket.on('connect', () => {
      console.log('🔗 Connected to GENESIS API');
      setSystemStatus('connected');
      setSocket(newSocket);
    });

    newSocket.on('disconnect', () => {
      console.log('❌ Disconnected from GENESIS API');
      setSystemStatus('disconnected');
    });

    newSocket.on('system_telemetry', (data) => {
      setTelemetryData(data);
    });

    newSocket.on('new_signal', (data) => {
      setSignals(prev => [data, ...prev.slice(0, 19)]);
    });

    newSocket.on('new_order', (data) => {
      setOrders(prev => [data, ...prev.slice(0, 19)]);
    });

    newSocket.on('kill_switch_activated', (data) => {
      setKillSwitchActive(true);
      console.log('🚨 KILL SWITCH ACTIVATED:', data);
    });

    newSocket.on('module_loaded', (data) => {
      console.log('✅ Module loaded:', data.module_name);
    });

    return () => newSocket.close();
  }, []);

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const connectionStatusColor = systemStatus === 'connected' ? 'success' : 'error';
  const connectionStatusText = systemStatus === 'connected' ? 'ONLINE' : 'OFFLINE';

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div className="App">
        {/* Header */}
        <AppBar position="static" sx={{ backgroundColor: '#000', borderBottom: '1px solid #333' }}>
          <Toolbar>
            <Typography variant="h4" component="div" sx={{ flexGrow: 1, fontFamily: '"Roboto Mono", monospace' }}>
              🚀 GENESIS v7.0
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Chip
                label={`ARCHITECT MODE • ${connectionStatusText}`}
                color={connectionStatusColor}
                variant="outlined"
                size="small"
              />
              {killSwitchActive && (
                <Chip
                  label="🚨 KILL SWITCH ACTIVE"
                  color="error"
                  variant="filled"
                  size="small"
                />
              )}
            </Box>
          </Toolbar>
        </AppBar>

        {/* Navigation Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', backgroundColor: '#111' }}>
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            sx={{
              '& .MuiTab-root': {
                color: '#b0b0b0',
                '&.Mui-selected': {
                  color: '#00ff88',
                },
              },
            }}
          >
            <Tab icon={<Settings />} label="System" />
            <Tab icon={<Timeline />} label="Telemetry" />
            <Tab icon={<TrendingUp />} label="Signals" />
            <Tab icon={<AccountBalance />} label="Trading" />
            <Tab icon={<Security />} label="Risk" />
            <Tab icon={<Speed />} label="Performance" />
          </Tabs>
        </Box>

        {/* Main Content */}
        <Container maxWidth="xl" sx={{ mt: 2 }}>
          {/* System Status Alert */}
          {systemStatus === 'disconnected' && (
            <Alert severity="error" sx={{ mb: 2 }}>
              🔴 Not connected to GENESIS API. Please check backend service.
            </Alert>
          )}

          {killSwitchActive && (
            <Alert severity="error" sx={{ mb: 2 }}>
              🚨 EMERGENCY: Kill Switch is ACTIVE. All trading operations are halted.
            </Alert>
          )}

          {/* Tab Panels */}
          <TabPanel value={currentTab} index={0}>
            <SystemPanel socket={socket} />
          </TabPanel>

          <TabPanel value={currentTab} index={1}>
            <TelemetryPanel telemetryData={telemetryData} socket={socket} />
          </TabPanel>

          <TabPanel value={currentTab} index={2}>
            <SignalPanel signals={signals} socket={socket} />
          </TabPanel>

          <TabPanel value={currentTab} index={3}>
            <TradingPanel orders={orders} socket={socket} />
          </TabPanel>

          <TabPanel value={currentTab} index={4}>
            <RiskPanel killSwitchActive={killSwitchActive} socket={socket} />
          </TabPanel>

          <TabPanel value={currentTab} index={5}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      📊 Performance Analytics
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Advanced performance metrics and analytics coming soon...
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>
        </Container>

        {/* Footer Status Bar */}
        <Box
          sx={{
            position: 'fixed',
            bottom: 0,
            left: 0,
            right: 0,
            backgroundColor: '#000',
            borderTop: '1px solid #333',
            p: 1,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <Typography variant="caption" color="text.secondary">
            GENESIS Trading Platform • Architect Mode v7.0 • Zero Tolerance Compliance
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {new Date().toLocaleString()}
          </Typography>
        </Box>
      </div>
    </ThemeProvider>
  );
}

export default App;
