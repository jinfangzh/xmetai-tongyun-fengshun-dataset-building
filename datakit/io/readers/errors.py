"""Errors raised by data-source discovery and decoding."""


class SourceDiscoveryError(Exception):
    """The requested source files could not be found."""

    def __init__(self, message: str, *, source_id: str = ""):
        super().__init__(message)
        self.source_id = source_id


class SourceDecodeError(Exception):
    """A source file could not be decoded."""

    def __init__(
        self, message: str, *, source_id: str = "", path: str = "", cause=None
    ):
        super().__init__(message)
        self.source_id = source_id
        self.path = path
        self.cause = cause
