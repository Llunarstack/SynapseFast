from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MissingOptionalDependencyError(ImportError):
    package: str
    extra: str
    hint: str

    def __str__(self) -> str:
        return (
            f"Missing optional dependency '{self.package}'. Install with: pip install -e \".[{self.extra}]\".\n"
            f"{self.hint}"
        )


def require(import_name: str, *, package: str, extra: str, hint: str):
    try:
        module = __import__(import_name)
        return module
    except Exception as e:
        raise MissingOptionalDependencyError(package=package, extra=extra, hint=hint) from e
