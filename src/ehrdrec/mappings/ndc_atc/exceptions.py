class NDCATCError(Exception):
    """Base exception for NDC to ATC mapping errors."""


class InvalidNDCError(NDCATCError):
    """Raised when an NDC cannot be normalised."""


class MappingStoreError(NDCATCError):
    """Raised when the mapping store is invalid or unreadable."""


class MappingNotFoundError(NDCATCError):
    """Raised when no mapping exists for a requested NDC."""