# 🛰️ GENESIS SYNC BEACON Documentation

## Overview

The GENESIS Sync Beacon is a real-time Google Drive monitoring system designed to track file changes in the "Genesis FINAL TRY" folder. It provides seamless integration with the GENESIS trading system's EventBus and telemetry infrastructure.

## Features

### 🔐 Security
- Service account authentication (no OAuth required)
- Read-only Google Drive access
- Encrypted credential storage
- No fallback or mock authentication

### 📡 Real-time Monitoring
- Continuous file scanning
- Change detection (new files, modifications)
- Real-time event emission
- Performance metrics tracking

### 🔗 GENESIS Integration
- Full EventBus wiring
- Comprehensive telemetry coverage
- Dashboard widget support
- System tree registration

### 📊 Telemetry & Monitoring
- File scan metrics
- Performance tracking
- Error monitoring
- Status reporting

## Installation

### Prerequisites

1. **Google Cloud Setup:**
   - Create a Google Cloud project
   - Enable Google Drive API
   - Create a service account
   - Download service account JSON key

2. **GENESIS System:**
   - GENESIS v7.0.0+ with Architect Mode
   - EventBus infrastructure
   - Telemetry system

### Installation Steps

1. **Run the installer:**
   ```bash
   python connectors/drive/install_sync_beacon.py
   ```

2. **Configure credentials:**
   ```bash
   # Replace the template with your service account JSON
   cp your_service_account.json connectors/drive/credentials/google_drive_service_account.json
   ```

3. **Test the connection:**
   ```bash
   python connectors/drive/genesis_sync_beacon.py
   ```

## Configuration

### Service Account Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Drive API
4. Create service account:
   - Go to IAM & Admin → Service Accounts
   - Click "Create Service Account"
   - Fill in details and create
   - Generate JSON key

5. Share the "Genesis FINAL TRY" folder with the service account email

### File Structure

```
connectors/drive/
├── genesis_sync_beacon.py              # Main beacon module
├── sync_beacon_eventbus_integration.py # EventBus configuration
├── install_sync_beacon.py              # Installation script
├── requirements.txt                    # Dependencies
├── README.md                          # This documentation
├── credentials/
│   ├── google_drive_service_account.json.template
│   └── google_drive_service_account.json  # Your actual credentials
└── logs/
    └── genesis_sync_report.txt         # Sync reports
```

## Usage

### Basic Usage

```python
from connectors.drive.genesis_sync_beacon import GENESISSyncBeacon

# Initialize beacon
beacon = GENESISSyncBeacon()

# Perform scan
results = beacon.scan_drive_files()

# Generate report
beacon.generate_sync_report(results)

# Get status
status = beacon.get_beacon_status()
```

### Command Line

```bash
# Run single scan
python connectors/drive/genesis_sync_beacon.py

# View last report
cat logs/genesis_sync_report.txt
```

## EventBus Integration

### Events Emitted

| Event | Description | Consumers |
|-------|-------------|-----------|
| `drive_sync_beacon_initializing` | Beacon startup | Dashboard, Telemetry |
| `drive_sync_started` | Scan operation started | Dashboard, Audit |
| `drive_sync_completed` | Scan operation completed | Dashboard, Notifications |
| `drive_sync_error` | Scan error occurred | Error Handler, Audit |
| `file_discovered` | New file detected | File Manager, Dashboard |
| `file_modified` | File modification detected | Sync Manager, Dashboard |
| `sync_beacon_status` | Status update | System Monitor, Health |

### Event Schema Examples

**drive_sync_completed:**
```json
{
  "module_id": "genesis_sync_beacon_abc123",
  "scan_id": "scan_1",
  "files_count": 42,
  "scan_duration_ms": 1250.5,
  "timestamp": "2025-06-22T10:30:00Z"
}
```

**file_discovered:**
```json
{
  "module_id": "genesis_sync_beacon_abc123",
  "file_info": {
    "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    "name": "trading_strategy.py",
    "modified_time": "2025-06-22T10:29:45Z",
    "size_bytes": "15420",
    "mime_type": "text/plain",
    "scan_timestamp": "2025-06-22T10:30:00Z"
  }
}
```

## Telemetry Metrics

### Performance Metrics

- **drive_files_scanned**: Number of files scanned per operation
- **scan_duration_ms**: Time taken for each scan operation
- **sync_beacon_initialized**: Initialization events
- **drive_authentication_success**: Authentication success events

### Error Tracking

- **drive_sync_error**: Scan operation errors
- **critical_error**: Critical system errors

### Status Monitoring

- **sync_beacon_status**: Periodic status updates
- **target_folder_discovered**: Folder discovery events

## Dashboard Widgets

### Status Card
- Real-time beacon status
- Authentication status
- Target folder information
- Last scan timestamp

### Performance Chart
- Files scanned over time
- Scan duration trends
- Error rate monitoring

### Activity Log
- Recent file discoveries
- File modifications
- Error events

## Security Considerations

### Credentials Security

1. **Never commit credentials to version control**
2. **Use read-only service account permissions**
3. **Regularly rotate service account keys**
4. **Monitor access logs**

### Access Control

1. **Limit Drive folder sharing to service account only**
2. **Use specific folder targeting (not root access)**
3. **Monitor unauthorized access attempts**

### Data Privacy

1. **File content is never downloaded or cached**
2. **Only metadata is accessed (name, timestamp, size)**
3. **All data transmission is encrypted**

## Troubleshooting

### Common Issues

**Authentication Failed:**
```
🚨 Service account file not found: connectors/drive/credentials/google_drive_service_account.json
```
- Solution: Ensure service account JSON is properly placed

**Folder Not Found:**
```
🚨 Target folder 'Genesis FINAL TRY' not found in Google Drive
```
- Solution: Verify folder name and service account access

**Permission Denied:**
```
🚨 Drive sync error: HttpError 403: Forbidden
```
- Solution: Share target folder with service account email

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export GENESIS_DEBUG=1
python connectors/drive/genesis_sync_beacon.py
```

### Log Analysis

Check logs for detailed information:
```bash
# Sync reports
tail -f logs/genesis_sync_report.txt

# System logs (if configured)
tail -f logs/genesis_sync_beacon.log
```

## API Reference

### GENESISSyncBeacon Class

#### Methods

**`__init__()`**
- Initializes the sync beacon with GENESIS compliance
- Sets up EventBus and telemetry connections
- Authenticates with Google Drive API

**`scan_drive_files() -> Dict[str, Any]`**
- Performs complete folder scan
- Detects new and modified files
- Emits appropriate events
- Returns scan results with metadata

**`generate_sync_report(scan_results: Dict[str, Any]) -> None`**
- Generates comprehensive sync report
- Saves to log file
- Outputs to console

**`get_beacon_status() -> Dict[str, Any]`**
- Returns current beacon status
- Includes performance metrics
- Emits status event

#### Properties

- `module_id`: Unique module identifier
- `version`: Beacon version
- `architect_mode`: Architect mode status
- `scan_count`: Number of scans performed
- `files_discovered`: Total files discovered
- `errors_encountered`: Error count

## Performance Optimization

### Scan Frequency

Recommended scan frequencies based on usage:

- **Development**: Every 5 minutes
- **Testing**: Every 2 minutes  
- **Production**: Every 10 minutes
- **Archive folders**: Every hour

### Large Folder Handling

For folders with 1000+ files:

1. Use pagination (automatically handled)
2. Implement incremental scanning
3. Cache file metadata
4. Monitor scan duration metrics

### Network Optimization

1. Use persistent HTTP connections
2. Implement retry logic with exponential backoff
3. Monitor API quota usage
4. Cache authentication tokens

## Integration Examples

### Dashboard Integration

```python
# Register dashboard widget
dashboard.register_widget({
    "id": "sync_beacon_status",
    "title": "🛰️ Drive Sync Beacon",
    "type": "status_card",
    "data_source": "sync_beacon_status"
})
```

### Notification Integration

```python
# Listen for new files
event_bus.subscribe("file_discovered", lambda event: 
    notify_new_file(event["file_info"]["name"])
)
```

### Backup Integration

```python
# Trigger backup on file changes
event_bus.subscribe("file_modified", lambda event:
    schedule_backup(event["file_info"]["id"])
)
```

## Compliance & Architecture

### ARCHITECT MODE v7.0.0 Compliance

✅ **Real-time data only** - No mock or simulated data  
✅ **EventBus integration** - All events properly routed  
✅ **Telemetry coverage** - Complete metrics tracking  
✅ **Error handling** - No fallback logic  
✅ **Module registration** - Registered in system tree  
✅ **Documentation** - Complete documentation  

### System Requirements

- **Python**: 3.8+
- **GENESIS**: v7.0.0+
- **Network**: HTTPS access to Google APIs
- **Storage**: 10MB minimum for logs and cache

## Support & Maintenance

### Monitoring

1. **Status checks**: Monitor beacon health via dashboard
2. **Performance metrics**: Track scan duration and file counts
3. **Error rates**: Monitor error frequency and types
4. **API usage**: Track Google Drive API quota consumption

### Maintenance Tasks

1. **Weekly**: Review error logs and performance metrics
2. **Monthly**: Rotate service account keys if required
3. **Quarterly**: Review and update folder permissions
4. **Yearly**: Archive old sync reports and logs

### Version Updates

The sync beacon follows semantic versioning:
- **Patch** (1.0.x): Bug fixes, security updates
- **Minor** (1.x.0): New features, EventBus changes
- **Major** (x.0.0): Breaking changes, API updates

---

**GENESIS Sync Beacon v1.0.0** - Real-time Google Drive monitoring for GENESIS Trading System  
**Architect Mode v7.0.0 Compliant** - No mocks, no fallbacks, real data only
