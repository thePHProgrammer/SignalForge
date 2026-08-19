"""Exception vocabulary shared by every data adapter."""


class DataAdapterError(Exception):
    """Base class for all data-adapter failures."""


class ApiRequestError(DataAdapterError):
    """The upstream API returned a non-200 or malformed response."""


class UnsupportedTimeframeError(DataAdapterError):
    """The requested timeframe can't be satisfied for this request."""


class SymbolNotFoundError(DataAdapterError):
    """The canonical symbol can't be mapped to a provider-specific identifier."""


class MissingCredentialsError(DataAdapterError):
    """A required credential (e.g. an API token) was not configured."""
