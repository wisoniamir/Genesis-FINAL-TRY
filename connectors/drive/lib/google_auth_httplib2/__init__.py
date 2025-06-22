"""Google Auth Library for httplib2.

This library provides integration between the httplib2 library and
google-auth.
"""

from __future__ import absolute_import

import httplib2

from google.auth import credentials

__version__ = "0.1.1"

class AuthorizedHttp:
    """A httplib2.Http-like object that is authorized with credentials.

    Args:
        credentials (google.auth.credentials.Credentials): The credentials to
            add to the request.
        http (httplib2.Http, optional): The underlying Http object to use to make
            requests. If not provided, a default Http object will be used.
        scopes (Sequence[str], optional): The list of scopes for the credentials.
            If not provided, the credentials' default scopes will be used.
        refresh_timeout (Optional[float]): The timeout value in seconds for
            credential refresh HTTP requests.
        **kwargs: Additional arguments passed through to the Http constructor.
    """

    def __init__(
        self,
        credentials,
        http=None,
        scopes=None,
        refresh_timeout=None,
        **kwargs
    ):
        """Initialize Authorized HTTP object."""
        self.credentials = credentials
        self.http = http or httplib2.Http(**kwargs)
        self.scopes = scopes
        self._refresh_timeout = refresh_timeout

    def request(
        self, uri, method="GET", body=None, headers=None, **kwargs
    ):
        """Make an HTTP request with authorized credentials.

        Args:
            uri (str): The URI to be requested.
            method (str): The HTTP method to use for the request. Defaults
                to 'GET'.
            body (Optional[bytes]): The payload / body in HTTP request.
            headers (Optional[Dict[str, str]]): Request headers.
            **kwargs: Additional arguments are passed through to the underlying
                :meth:`~httplib2.Http.request` method.

        Returns:
            Tuple[Dict[str, Any], bytes]: The response headers and payload. 
        """
        headers = headers if headers is not None else {}

        # Add the token to the headers
        if self.credentials:
            if self.scopes:
                self.credentials = self.credentials.with_scopes(self.scopes)

            if self.credentials.expired:
                self.credentials.refresh()

            self.credentials.apply(headers)

        return self.http.request(uri, method, body=body, headers=headers, **kwargs)

    def close(self):
        """Close the underlying Http connection."""
        self.http.close()
