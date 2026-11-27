"""Discovery and isolated loading for installed Maya MCP extensions."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from importlib import metadata
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from fastmcp import FastMCP

ENTRY_POINT_GROUP = "grandakun.maya_mcp.extensions"
DISABLED_EXTENSIONS_ENV_VAR = "DCC_MAYA_DISABLED_EXTENSIONS"
_EXTENSION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_LOGGER = logging.getLogger(__name__)


class ExtensionEntryPoint(Protocol):
    """Entry Point surface used by the loader and its test doubles."""

    name: str
    value: str

    def load(self) -> object:
        """Load and return the configured extension factory."""


@dataclass(frozen=True)
class MayaMcpExtension:
    """Declarative contract implemented by a Maya MCP extension package."""

    id: str
    version: str
    tool_names: tuple[str, ...]
    register: Callable[[FastMCP], None]


@dataclass(frozen=True)
class ExtensionLoadResult:
    """Safe startup result for one discovered extension."""

    entry_point: str
    extension_id: str
    status: Literal["loaded", "disabled", "failed"]
    version: str | None = None
    tool_names: tuple[str, ...] = ()
    error: str | None = None


def discover_extensions() -> tuple[metadata.EntryPoint, ...]:
    """Return installed Maya MCP extension Entry Points in deterministic order."""
    entry_points = metadata.entry_points(group=ENTRY_POINT_GROUP)
    return tuple(sorted(entry_points, key=lambda item: (item.name, item.value)))


def _disabled_extension_ids(value: Iterable[str] | None) -> frozenset[str]:
    if value is None:
        value = os.environ.get(DISABLED_EXTENSIONS_ENV_VAR, "").split(",")
    return frozenset(item.strip() for item in value if item.strip())


def _failure(
    entry_point: ExtensionEntryPoint,
    extension_id: str,
    error: str,
) -> ExtensionLoadResult:
    result = ExtensionLoadResult(
        entry_point=entry_point.value,
        extension_id=extension_id,
        status="failed",
        error=error,
    )
    _LOGGER.warning("Maya MCP extension %s failed: %s", extension_id, error)
    return result


def _component_store(mcp: FastMCP) -> dict[str, Any]:
    """Return FastMCP's local component store for atomic startup registration."""
    provider = getattr(mcp, "_local_provider", None)
    components = getattr(provider, "_components", None)
    if not isinstance(components, dict):
        raise RuntimeError("FastMCP local component store is unavailable")
    return cast("dict[str, Any]", components)


def _tool_map(components: dict[str, Any]) -> dict[str, Any]:
    return {
        component.name: component
        for identity, component in components.items()
        if identity.startswith("tool:")
    }


def _restore_components(mcp: FastMCP, snapshot: dict[str, Any]) -> None:
    components = _component_store(mcp)
    components.clear()
    components.update(snapshot)


def load_extensions(
    mcp: FastMCP,
    *,
    entry_points: Iterable[ExtensionEntryPoint] | None = None,
    disabled_extensions: Iterable[str] | None = None,
) -> tuple[ExtensionLoadResult, ...]:
    """Load installed extensions without allowing one failure to stop the server."""
    if entry_points is None:
        discovered = cast("tuple[ExtensionEntryPoint, ...]", discover_extensions())
    else:
        discovered = tuple(entry_points)
    ordered = sorted(discovered, key=lambda item: (item.name, item.value))
    disabled = _disabled_extension_ids(disabled_extensions)
    seen_extension_ids: set[str] = set()
    results: list[ExtensionLoadResult] = []

    for entry_point in ordered:
        if entry_point.name in disabled:
            results.append(
                ExtensionLoadResult(
                    entry_point=entry_point.value,
                    extension_id=entry_point.name,
                    status="disabled",
                )
            )
            continue

        try:
            factory = entry_point.load()
            if not callable(factory):
                raise TypeError("extension Entry Point must resolve to a callable")
            extension = factory()
        except Exception as exc:  # Extension import failures are intentionally isolated.
            results.append(_failure(entry_point, entry_point.name, f"load: {type(exc).__name__}"))
            continue

        if not isinstance(extension, MayaMcpExtension):
            results.append(
                _failure(entry_point, entry_point.name, "validate: invalid extension descriptor")
            )
            continue
        if not _EXTENSION_ID_PATTERN.fullmatch(extension.id):
            results.append(_failure(entry_point, extension.id, "validate: invalid extension id"))
            continue
        if entry_point.name != extension.id:
            results.append(
                _failure(
                    entry_point,
                    extension.id,
                    "validate: entry point name must match extension id",
                )
            )
            continue
        if extension.id in seen_extension_ids:
            results.append(_failure(entry_point, extension.id, "validate: duplicate extension id"))
            continue
        seen_extension_ids.add(extension.id)

        if (
            not isinstance(extension.tool_names, tuple)
            or any(not isinstance(name, str) or not name for name in extension.tool_names)
            or len(set(extension.tool_names)) != len(extension.tool_names)
        ):
            results.append(_failure(entry_point, extension.id, "validate: duplicate tool name"))
            continue

        try:
            snapshot = _component_store(mcp).copy()
        except RuntimeError as exc:
            results.append(_failure(entry_point, extension.id, f"register: {type(exc).__name__}"))
            continue
        before = _tool_map(snapshot)
        if set(extension.tool_names).intersection(before):
            results.append(_failure(entry_point, extension.id, "validate: duplicate tool name"))
            continue

        try:
            extension.register(mcp)
        except Exception as exc:  # Extension registration failures are intentionally isolated.
            _restore_components(mcp, snapshot)
            results.append(_failure(entry_point, extension.id, f"register: {type(exc).__name__}"))
            continue

        after = _tool_map(_component_store(mcp))
        registered_names = set(after).difference(before)
        existing_tools_changed = any(after.get(name) is not tool for name, tool in before.items())
        if registered_names != set(extension.tool_names) or existing_tools_changed:
            _restore_components(mcp, snapshot)
            results.append(
                _failure(
                    entry_point,
                    extension.id,
                    "validate: registered tools do not match declaration",
                )
            )
            continue

        results.append(
            ExtensionLoadResult(
                entry_point=entry_point.value,
                extension_id=extension.id,
                status="loaded",
                version=extension.version,
                tool_names=extension.tool_names,
            )
        )

    return tuple(results)
