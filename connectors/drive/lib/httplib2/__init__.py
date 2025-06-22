
# Minimal httplib2 implementation for GENESIS Sync Beacon
import http.client

class Http:
    def __init__(self, **kwargs):
        self.connections = {}
    
    def request(self, uri, method="GET", body=None, headers=None, **kwargs):
        from urllib.parse import urlparse
        headers = headers or {}
        parts = urlparse(uri)
        if parts.scheme == 'https':
            conn = http.client.HTTPSConnection(parts.netloc)
        else:
            conn = http.client.HTTPConnection(parts.netloc)
        
        path = parts.path
        if parts.query:
            path += '?' + parts.query
            
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        
        return ({
            'status': resp.status,
            'reason': resp.reason
        }, resp.read())
