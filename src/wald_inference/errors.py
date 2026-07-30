"""Domain-specific errors for Wald inference."""


class ValidationError(ValueError):
    """Raised when inputs cannot support the requested Wald calculation."""
