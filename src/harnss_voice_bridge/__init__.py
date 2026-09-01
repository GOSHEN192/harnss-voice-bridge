"""Minimal, isolated bridge between voice services and Harnss Remote Control."""

__version__ = "0.1.0"


def main() -> int:
    """Run the command-line interface without importing it during package load."""
    from .__main__ import main as cli_main

    return cli_main()