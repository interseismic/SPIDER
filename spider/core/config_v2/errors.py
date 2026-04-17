"""
Structured error and warning types for config_v2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional


IssueSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class ConfigIssue:
    path: str
    code: str
    message: str
    severity: IssueSeverity = "error"
    suggestion: Optional[str] = None

    def render(self) -> str:
        base = f"[{self.severity}] {self.path} ({self.code}): {self.message}"
        if self.suggestion:
            return f"{base} | suggestion: {self.suggestion}"
        return base


class ConfigError(ValueError):
    def __init__(self, issues: Iterable[ConfigIssue]):
        self.issues: List[ConfigIssue] = list(issues)
        if not self.issues:
            self.issues = [ConfigIssue(path="<root>", code="unknown", message="Unknown configuration error")]
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return "Configuration validation failed:\n" + "\n".join(
            f"- {issue.render()}" for issue in self.issues
        )


def single_issue_error(
    *,
    path: str,
    code: str,
    message: str,
    suggestion: Optional[str] = None,
) -> ConfigError:
    return ConfigError(
        [
            ConfigIssue(
                path=path,
                code=code,
                message=message,
                severity="error",
                suggestion=suggestion,
            )
        ]
    )

