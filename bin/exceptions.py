class NoExtensionError(Exception):
    """
    Raised when no extension in path present but is required.
    """


class ExpError(Exception):
    ...


class InitialPopFixError(Exception):
    """
    Raised when cannot fix or regenerate and fix random solutions.
    """
