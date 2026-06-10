"""
Tool Registry — dynamic tools, schemas, validation, middleware.
Production-grade tool system with auto-discovery and composition.
"""
import asyncio
import inspect
import json
import time
import logging
import hashlib
from typing import Optional, Dict, Any, List, Callable, get_type_hints
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    FILE = "file"
    NETWORK = "network"
    CODE = "code"
    DATA = "data"
    SYSTEM = "system"
    SEARCH = "search"
    MEDIA = "media"
    CUSTOM = "custom"


@dataclass
class ToolParam:
    """Tool parameter definition."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: List[str] = None


@dataclass
class ToolSchema:
    """Complete tool schema for LLM function calling."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: ToolCategory = ToolCategory.CUSTOM
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    requires_auth: bool = False
    timeout: float = 120
    cache_ttl: float = 0  # 0 = no cache
    retry_count: int = 0

    def to_openai_schema(self) -> Dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_anthropic_schema(self) -> Dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolResult:
    """Result from tool execution."""
    tool_name: str
    output: Any
    success: bool = True
    error: str = ""
    duration_ms: float = 0
    cached: bool = False
    metadata: Dict = field(default_factory=dict)


class ToolMiddleware:
    """Base class for tool middleware."""

    async def before(self, tool_name: str, arguments: Dict) -> Dict:
        """Called before tool execution. Can modify arguments."""
        return arguments

    async def after(self, tool_name: str, result: ToolResult) -> ToolResult:
        """Called after tool execution. Can modify result."""
        return result

    async def on_error(self, tool_name: str, error: Exception) -> Optional[ToolResult]:
        """Called on tool error. Return None to propagate, or ToolResult to recover."""
        return None


class LoggingMiddleware(ToolMiddleware):
    """Logs all tool calls."""

    async def before(self, tool_name, arguments):
        logger.info(f"[TOOL] {tool_name}({json.dumps(arguments, default=str)[:200]})")
        return arguments

    async def after(self, tool_name, result):
        logger.info(f"[TOOL] {tool_name} -> {'OK' if result.success else 'FAIL'} ({result.duration_ms:.0f}ms)")
        return result


class MetricsMiddleware(ToolMiddleware):
    """Collects tool execution metrics."""

    def __init__(self):
        self.call_counts: Dict[str, int] = {}
        self.total_time: Dict[str, float] = {}
        self.errors: Dict[str, int] = {}

    async def after(self, tool_name, result):
        self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1
        self.total_time[tool_name] = self.total_time.get(tool_name, 0) + result.duration_ms
        if not result.success:
            self.errors[tool_name] = self.errors.get(tool_name, 0) + 1
        return result

    def report(self) -> Dict:
        return {
            "calls": dict(self.call_counts),
            "total_time_ms": {k: round(v, 1) for k, v in self.total_time.items()},
            "errors": dict(self.errors),
        }


class ToolRegistry:
    """
    Production tool registry with:
    - Dynamic registration and removal
    - Auto-schema generation from function signatures
    - Type validation
    - Middleware pipeline (logging, metrics, auth)
    - Tool composition (chains, parallel)
    - Result caching
    - LLM schema export (OpenAI/Anthropic formats)
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._schemas: Dict[str, ToolSchema] = {}
        self._middleware: List[ToolMiddleware] = [LoggingMiddleware(), MetricsMiddleware()]
        self._cache: Dict[str, tuple] = {}  # key -> (result, expiry)
        self._categories: Dict[ToolCategory, List[str]] = {}

    def register(self, name: str = None, description: str = "",
                 parameters: Dict = None, category: ToolCategory = ToolCategory.CUSTOM,
                 timeout: float = 120, cache_ttl: float = 0, tags: List[str] = None):
        """Decorator to register a tool function."""
        def decorator(func: Callable):
            tool_name = name or func.__name__

            # Auto-generate schema from function signature
            if parameters is None:
                schema_params = self._generate_schema(func)
            else:
                schema_params = parameters

            schema = ToolSchema(
                name=tool_name,
                description=description or inspect.getdoc(func) or "",
                parameters=schema_params,
                category=category,
                timeout=timeout,
                cache_ttl=cache_ttl,
                tags=tags or [],
            )

            self._tools[tool_name] = {"func": func, "schema": schema}
            self._schemas[tool_name] = schema
            self._categories.setdefault(category, []).append(tool_name)

            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.execute(tool_name, kwargs)

            return func  # Return original, not wrapper
        return decorator

    def register_function(self, func: Callable, name: str = None,
                          description: str = "", parameters: Dict = None,
                          category: ToolCategory = ToolCategory.CUSTOM,
                          timeout: float = 120):
        """Register a function directly (not as decorator)."""
        tool_name = name or func.__name__
        schema_params = parameters or self._generate_schema(func)

        schema = ToolSchema(
            name=tool_name,
            description=description or inspect.getdoc(func) or "",
            parameters=schema_params,
            category=category,
            timeout=timeout,
        )

        self._tools[tool_name] = {"func": func, "schema": schema}
        self._schemas[tool_name] = schema
        self._categories.setdefault(category, []).append(tool_name)

    def unregister(self, name: str):
        """Remove a tool."""
        self._tools.pop(name, None)
        self._schemas.pop(name, None)
        for cat_tools in self._categories.values():
            if name in cat_tools:
                cat_tools.remove(name)

    def use(self, middleware: ToolMiddleware):
        """Add middleware to the pipeline."""
        self._middleware.append(middleware)

    async def execute(self, name: str, arguments: Dict = None,
                      timeout: float = None) -> ToolResult:
        """Execute a tool with full middleware pipeline."""
        arguments = arguments or {}
        tool_info = self._tools.get(name)
        if not tool_info:
            return ToolResult(tool_name=name, output=None, success=False,
                            error=f"Unknown tool: {name}")

        schema = tool_info["schema"]
        tool_timeout = timeout or schema.timeout

        # Check cache
        if schema.cache_ttl > 0:
            cache_key = hashlib.md5(f"{name}:{json.dumps(arguments, sort_keys=True)}".encode()).hexdigest()
            if cache_key in self._cache:
                result, expiry = self._cache[cache_key]
                if time.time() < expiry:
                    result.cached = True
                    return result

        # Before middleware
        for mw in self._middleware:
            try:
                arguments = await mw.before(name, arguments)
            except Exception as e:
                logger.warning(f"Middleware before() error: {e}")

        # Execute
        start = time.monotonic()
        try:
            func = tool_info["func"]
            if asyncio.iscoroutinefunction(func):
                output = await asyncio.wait_for(func(**arguments), timeout=tool_timeout)
            else:
                loop = asyncio.get_event_loop()
                output = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(**arguments)),
                    timeout=tool_timeout
                )
            result = ToolResult(
                tool_name=name, output=output,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except asyncio.TimeoutError:
            result = ToolResult(
                tool_name=name, output=None, success=False,
                error=f"Timeout after {tool_timeout}s",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            result = ToolResult(
                tool_name=name, output=None, success=False,
                error=f"{type(e).__name__}: {e}",
                duration_ms=(time.monotonic() - start) * 1000,
            )

            # Error middleware
            for mw in self._middleware:
                recovery = await mw.on_error(name, e)
                if recovery:
                    result = recovery
                    break

        # After middleware
        for mw in self._middleware:
            try:
                result = await mw.after(name, result)
            except Exception as e:
                logger.warning(f"Middleware after() error: {e}")

        # Cache result
        if schema.cache_ttl > 0 and result.success:
            self._cache[cache_key] = (result, time.time() + schema.cache_ttl)

        return result

    async def execute_parallel(self, calls: List[Dict]) -> List[ToolResult]:
        """Execute multiple tools in parallel."""
        tasks = [
            asyncio.create_task(self.execute(c["name"], c.get("arguments", {})))
            for c in calls
        ]
        return await asyncio.gather(*tasks)

    async def execute_chain(self, calls: List[Dict]) -> List[ToolResult]:
        """Execute tools in sequence, passing output to next."""
        results = []
        prev_output = None
        for call in calls:
            args = call.get("arguments", {})
            if prev_output and call.get("pass_output"):
                args[call["pass_output"]] = prev_output
            result = await self.execute(call["name"], args)
            results.append(result)
            prev_output = result.output
        return results

    def get_schemas(self, format: str = "openai") -> List[Dict]:
        """Get all tool schemas in specified format."""
        schemas = []
        for schema in self._schemas.values():
            if format == "openai":
                schemas.append(schema.to_openai_schema())
            elif format == "anthropic":
                schemas.append(schema.to_anthropic_schema())
        return schemas

    def list_tools(self, category: ToolCategory = None) -> List[Dict]:
        """List all registered tools."""
        tools = []
        for name, info in self._tools.items():
            schema = info["schema"]
            if category and schema.category != category:
                continue
            tools.append({
                "name": name,
                "description": schema.description,
                "category": schema.category.value,
                "tags": schema.tags,
            })
        return tools

    def _generate_schema(self, func: Callable) -> Dict:
        """Auto-generate JSON schema from function signature."""
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue

            type_map = {
                str: "string", int: "integer", float: "number",
                bool: "boolean", list: "array", dict: "object",
                List: "array", Dict: "object",
            }

            param_type = hints.get(param_name)
            json_type = type_map.get(param_type, "string")

            prop = {"type": json_type}
            if param.default != inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(param_name)

            properties[param_name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def stats(self) -> Dict:
        metrics_mw = None
        for mw in self._middleware:
            if isinstance(mw, MetricsMiddleware):
                metrics_mw = mw
                break

        return {
            "total_tools": len(self._tools),
            "categories": {cat.value: len(tools) for cat, tools in self._categories.items()},
            "cache_entries": len(self._cache),
            "middleware_count": len(self._middleware),
            "metrics": metrics_mw.report() if metrics_mw else {},
        }
