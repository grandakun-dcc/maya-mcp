"""Tests for installed Maya MCP extension discovery and isolation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from maya_mcp.extensions import MayaMcpExtension, load_extensions

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class FakeEntryPoint:
    """Minimal importlib.metadata.EntryPoint test double."""

    name: str
    value: str
    loader: Callable[[], Any]

    def load(self) -> Any:
        return self.loader()


def _extension(
    extension_id: str,
    tool_names: tuple[str, ...],
    register: Callable[[FastMCP], None],
) -> MayaMcpExtension:
    return MayaMcpExtension(
        id=extension_id,
        version="1.0.0",
        tool_names=tool_names,
        register=register,
    )


def _entry_point(
    name: str,
    extension_factory: Callable[[], MayaMcpExtension],
) -> FakeEntryPoint:
    return FakeEntryPoint(
        name=name,
        value=f"tests.fake_{name}:get_extension",
        loader=lambda: extension_factory,
    )


def _register_named_tool(name: str) -> Callable[[FastMCP], None]:
    def register(mcp: FastMCP) -> None:
        @mcp.tool(name=name)
        def extension_tool() -> str:
            return name

    return register


def _tool_names(mcp: FastMCP) -> tuple[str, ...]:
    return tuple(tool.name for tool in asyncio.run(mcp.list_tools()))


def test_load_extensions_sorts_entry_points_by_name() -> None:
    mcp = FastMCP("test")
    calls: list[str] = []

    def make(name: str) -> MayaMcpExtension:
        def register(server: FastMCP) -> None:
            calls.append(name)
            _register_named_tool(f"{name}_tool")(server)

        return _extension(name, (f"{name}_tool",), register)

    results = load_extensions(
        mcp,
        entry_points=(
            _entry_point("xiaota", lambda: make("xiaota")),
            _entry_point("hairpin", lambda: make("hairpin")),
        ),
        disabled_extensions=(),
    )

    assert calls == ["hairpin", "xiaota"]
    assert [result.extension_id for result in results] == ["hairpin", "xiaota"]
    assert all(result.status == "loaded" for result in results)


def test_disabled_extension_is_not_imported() -> None:
    mcp = FastMCP("test")

    def fail_if_loaded() -> Any:
        raise AssertionError("disabled extension should not be imported")

    results = load_extensions(
        mcp,
        entry_points=(FakeEntryPoint("hairpin", "tests.fake:factory", fail_if_loaded),),
        disabled_extensions=(" hairpin ", "hairpin", "unknown"),
    )

    assert results[0].status == "disabled"
    assert results[0].extension_id == "hairpin"
    assert _tool_names(mcp) == ()


def test_import_failure_does_not_stop_later_extensions_or_expose_message() -> None:
    mcp = FastMCP("test")

    def fail_import() -> Any:
        raise RuntimeError("secret-token-value")

    good = _entry_point(
        "xiaota",
        lambda: _extension("xiaota", ("xiaota_tool",), _register_named_tool("xiaota_tool")),
    )
    results = load_extensions(
        mcp,
        entry_points=(FakeEntryPoint("broken", "tests.broken:factory", fail_import), good),
        disabled_extensions=(),
    )

    assert [result.status for result in results] == ["failed", "loaded"]
    assert results[0].error == "load: RuntimeError"
    assert "secret-token-value" not in repr(results)
    assert _tool_names(mcp) == ("xiaota_tool",)


def test_duplicate_extension_id_is_rejected() -> None:
    mcp = FastMCP("test")
    first = _entry_point(
        "xiaota",
        lambda: _extension("xiaota", ("first_tool",), _register_named_tool("first_tool")),
    )
    second = _entry_point(
        "xiaota",
        lambda: _extension("xiaota", ("second_tool",), _register_named_tool("second_tool")),
    )

    results = load_extensions(
        mcp,
        entry_points=(first, second),
        disabled_extensions=(),
    )

    assert [result.status for result in results] == ["loaded", "failed"]
    assert results[1].error == "validate: duplicate extension id"
    assert _tool_names(mcp) == ("first_tool",)


def test_declared_tool_collision_is_rejected_before_registration() -> None:
    mcp = FastMCP("test")
    _register_named_tool("shared_tool")(mcp)
    called = False

    def register(server: FastMCP) -> None:
        nonlocal called
        called = True
        _register_named_tool("shared_tool")(server)

    entry_point = _entry_point(
        "hairpin",
        lambda: _extension("hairpin", ("shared_tool",), register),
    )

    results = load_extensions(mcp, entry_points=(entry_point,), disabled_extensions=())

    assert results[0].status == "failed"
    assert results[0].error == "validate: duplicate tool name"
    assert called is False
    assert _tool_names(mcp) == ("shared_tool",)


def test_partial_registration_failure_rolls_back_extension_tools() -> None:
    mcp = FastMCP("test")
    _register_named_tool("base_tool")(mcp)

    def register(server: FastMCP) -> None:
        _register_named_tool("partial_tool")(server)
        raise ValueError("failed after registration")

    entry_point = _entry_point(
        "hairpin",
        lambda: _extension("hairpin", ("partial_tool",), register),
    )

    results = load_extensions(mcp, entry_points=(entry_point,), disabled_extensions=())

    assert results[0].status == "failed"
    assert results[0].error == "register: ValueError"
    assert _tool_names(mcp) == ("base_tool",)


def test_declared_and_registered_tools_must_match() -> None:
    mcp = FastMCP("test")
    entry_point = _entry_point(
        "hairpin",
        lambda: _extension(
            "hairpin",
            ("declared_tool",),
            _register_named_tool("unexpected_tool"),
        ),
    )

    results = load_extensions(mcp, entry_points=(entry_point,), disabled_extensions=())

    assert results[0].status == "failed"
    assert results[0].error == "validate: registered tools do not match declaration"
    assert _tool_names(mcp) == ()


def test_entry_point_name_must_match_extension_id() -> None:
    mcp = FastMCP("test")
    entry_point = _entry_point(
        "hairpin",
        lambda: _extension("different_id", (), lambda _server: None),
    )

    results = load_extensions(mcp, entry_points=(entry_point,), disabled_extensions=())

    assert results[0].status == "failed"
    assert results[0].error == "validate: entry point name must match extension id"


def test_create_server_loads_extensions_after_base_tools(monkeypatch: Any) -> None:
    from maya_mcp import server

    observed_tool_names: tuple[str, ...] = ()

    def fake_load_extensions(mcp: FastMCP) -> tuple[()]:
        nonlocal observed_tool_names
        observed_tool_names = _tool_names(mcp)
        return ()

    monkeypatch.setattr(server, "load_extensions", fake_load_extensions)

    server.create_server()

    assert len(observed_tool_names) == 71


def test_create_server_can_skip_installed_extensions(monkeypatch: Any) -> None:
    from maya_mcp import server

    def fail_if_called(_mcp: FastMCP) -> tuple[()]:
        raise AssertionError("installed extensions must not be loaded")

    monkeypatch.setattr(server, "load_extensions", fail_if_called)

    mcp = server.create_server(load_installed_extensions=False)

    assert len(_tool_names(mcp)) == 71
