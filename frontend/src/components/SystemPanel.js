/**
 * 🔧 System Panel Component
 * Real-time system monitoring and module management
 */

import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Box,
  Alert,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  TextField,
  DialogActions
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Refresh,
  Warning,
  CheckCircle,
  Error,
  Link,
  LinkOff
} from '@mui/icons-material';
import axios from 'axios';

const SystemPanel = ({ socket }) => {
  const [modules, setModules] = useState([]);
  const [systemStats, setSystemStats] = useState({});
  const [mt5Connected, setMt5Connected] = useState(false);
  const [mt5Dialog, setMt5Dialog] = useState(false);
  const [mt5Credentials, setMt5Credentials] = useState({
    login: '',
    password: '',
    server: ''
  });

  useEffect(() => {
    loadModules();
    loadSystemStatus();

    const interval = setInterval(() => {
      loadModules();
      loadSystemStatus();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const loadModules = async () => {
    try {
      const response = await axios.get('/api/modules');
      setModules(response.data);
    } catch (error) {
      console.error('Failed to load modules:', error);
    }
  };

  const loadSystemStatus = async () => {
    try {
      const response = await axios.get('/api/system/status');
      setSystemStats(response.data);
      setMt5Connected(response.data.mt5_connected || false);
    } catch (error) {
      console.error('Failed to load system status:', error);
    }
  };

  const loadModule = async (moduleName) => {
    try {
      await axios.post('/api/modules/load', { module_name: moduleName });
      loadModules();
    } catch (error) {
      console.error(`Failed to load module ${moduleName}:`, error);
    }
  };

  const connectMT5 = async () => {
    try {
      await axios.post('/api/mt5/connect', mt5Credentials);
      setMt5Connected(true);
      setMt5Dialog(false);
      setMt5Credentials({ login: '', password: '', server: '' });
    } catch (error) {
      console.error('MT5 connection failed:', error);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'operational':
      case 'active':
        return <CheckCircle color="success" />;
      case 'error':
      case 'stopped':
        return <Error color="error" />;
      default:
        return <Warning color="warning" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'operational':
      case 'active':
        return 'success';
      case 'error':
      case 'stopped':
        return 'error';
      default:
        return 'warning';
    }
  };

  const getComplianceColor = (score) => {
    if (score >= 9) return 'success';
    if (score >= 7) return 'warning';
    return 'error';
  };

  return (
    <Grid container spacing={3}>
      {/* System Overview */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              🔧 System Overview
            </Typography>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Status: {systemStats.status || 'Unknown'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Modules Loaded: {modules.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Last Update: {systemStats.last_update ? new Date(systemStats.last_update).toLocaleTimeString() : 'N/A'}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                startIcon={<PlayArrow />}
                size="small"
                color="success"
              >
                Start System
              </Button>
              <Button
                variant="outlined"
                startIcon={<Stop />}
                size="small"
                color="error"
              >
                Stop System
              </Button>
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                size="small"
                onClick={loadModules}
              >
                Refresh
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* MT5 Connection */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📡 MT5 Connection
            </Typography>
            <Box sx={{ mb: 2 }}>
              <Chip
                icon={mt5Connected ? <Link /> : <LinkOff />}
                label={mt5Connected ? 'Connected' : 'Disconnected'}
                color={mt5Connected ? 'success' : 'error'}
                variant="outlined"
              />
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="contained"
                onClick={() => setMt5Dialog(true)}
                disabled={mt5Connected}
                size="small"
              >
                Connect MT5
              </Button>
              <Button
                variant="outlined"
                color="error"
                disabled={!mt5Connected}
                size="small"
              >
                Disconnect
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* Module Status */}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📊 Module Status
            </Typography>
            {modules.length === 0 && (
              <Alert severity="info">
                No modules loaded. Click the buttons below to load core modules.
              </Alert>
            )}
            <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Button
                variant="outlined"
                size="small"
                onClick={() => loadModule('strategy_engine')}
              >
                Load Strategy Engine
              </Button>
              <Button
                variant="outlined"
                size="small"
                onClick={() => loadModule('execution_manager')}
              >
                Load Execution Manager
              </Button>
              <Button
                variant="outlined"
                size="small"
                onClick={() => loadModule('kill_switch')}
              >
                Load Kill Switch
              </Button>
            </Box>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Module</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Compliance</TableCell>
                    <TableCell>Last Update</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {modules.map((module) => (
                    <TableRow key={module.module_name}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {getStatusIcon(module.status)}
                          {module.module_name}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={module.status}
                          color={getStatusColor(module.status)}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Chip
                            label={`${module.compliance_score}/10`}
                            color={getComplianceColor(module.compliance_score)}
                            size="small"
                          />
                          <LinearProgress
                            variant="determinate"
                            value={(module.compliance_score / 10) * 100}
                            sx={{ width: 50, height: 4 }}
                            color={getComplianceColor(module.compliance_score)}
                          />
                        </Box>
                      </TableCell>
                      <TableCell>
                        {module.last_update ? new Date(module.last_update).toLocaleTimeString() : 'N/A'}
                      </TableCell>
                      <TableCell>
                        <Button size="small" variant="outlined">
                          Reload
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* MT5 Connection Dialog */}
      <Dialog open={mt5Dialog} onClose={() => setMt5Dialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>🔗 Connect to MetaTrader 5</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <TextField
              fullWidth
              label="Login"
              value={mt5Credentials.login}
              onChange={(e) => setMt5Credentials(prev => ({ ...prev, login: e.target.value }))}
              margin="normal"
              type="number"
            />
            <TextField
              fullWidth
              label="Password"
              type="password"
              value={mt5Credentials.password}
              onChange={(e) => setMt5Credentials(prev => ({ ...prev, password: e.target.value }))}
              margin="normal"
            />
            <TextField
              fullWidth
              label="Server"
              value={mt5Credentials.server}
              onChange={(e) => setMt5Credentials(prev => ({ ...prev, server: e.target.value }))}
              margin="normal"
              placeholder="e.g., MetaQuotes-Demo"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMt5Dialog(false)}>Cancel</Button>
          <Button onClick={connectMT5} variant="contained">Connect</Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
};

export default SystemPanel;
