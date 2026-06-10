"""
Plugin Engine — hot-loadable plugins with lifecycle hooks.
Supports: load, unload, reload, enable, disable.
"""
import importlib
import importlib.util
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PluginBase(ABC):
    """Base class all plugins must extend."""

    name: str = "unnamed"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    dependencies: List[str] = []

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = True
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Called when plugin is loaded. Return True if ready."""
        return True

    async def shutdown(self):
        """Called when plugin is unloaded."""
        pass

    async def on_event(self, event: str, data: Dict[str, Any]):
        """Handle events from the event bus."""
        pass

    def get_tools(self) -> Dict[str, Callable]:
        """Return tool functions this plugin provides."""
        return {}

    def get_hooks(self) -> Dict[str, Callable]:
        """Return lifecycle hooks this plugin provides."""
        return {}


@dataclass
class PluginInfo:
    """Metadata about a loaded plugin."""
    name: str
    version: str
    description: str
    path: str
    instance: Optional[PluginBase] = None
    enabled: bool = True
    loaded_at: float = field(default_factory=time.time)
    error: Optional[str] = None


class PluginEngine:
    """
    Plugin management with:
    - Hot-loading from directory
    - Dependency resolution
    - Lifecycle management
    - Tool registration
    - Event routing
    - Error isolation (one bad plugin won't crash others)
    """

    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or []
        self._plugins: Dict[str, PluginInfo] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._tools: Dict[str, Callable] = {}

    async def discover(self) -> List[str]:
        """Discover all plugins in configured directories."""
        discovered = []
        for dir_path in self.plugin_dirs:
            path = Path(dir_path)
            if not path.exists():
                continue
            for item in path.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    discovered.append(str(item))
                elif item.suffix == ".py" and item.stem != "__init__":
                    discovered.append(str(item))
        return discovered

    async def load(self, plugin_path: str, config: Dict = None) -> bool:
        """Load a plugin from path."""
        try:
            path = Path(plugin_path)
            plugin_name = path.stem if path.is_file() else path.name

            # Import the module
            if path.is_file():
                spec = importlib.util.spec_from_file_location(plugin_name, str(path))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                # Add parent to sys.path temporarily
                parent = str(path.parent)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                module = importlib.import_module(plugin_name)
                sys.path.remove(parent)

            # Find PluginBase subclass
            plugin_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, PluginBase) and obj is not PluginBase:
                    plugin_class = obj
                    break

            if not plugin_class:
                logger.warning(f"No PluginBase subclass found in {plugin_path}")
                return False

            # Check dependencies
            for dep in plugin_class.dependencies:
                if dep not in self._plugins:
                    logger.error(f"Plugin {plugin_name} requires {dep} which is not loaded")
                    return False

            # Instantiate and initialize
            instance = plugin_class(config)
            success = await instance.initialize()
            if not success:
                logger.error(f"Plugin {plugin_name} failed to initialize")
                return False

            # Register
            info = PluginInfo(
                name=plugin_name,
                version=plugin_class.version,
                description=plugin_class.description,
                path=plugin_path,
                instance=instance,
            )
            self._plugins[plugin_name] = info

            # Register tools
            for tool_name, tool_func in instance.get_tools().items():
                self._tools[f"{plugin_name}.{tool_name}"] = tool_func
                self._tools[tool_name] = tool_func  # Also register without prefix

            # Register hooks
            for hook_name, hook_func in instance.get_hooks().items():
                self._hooks.setdefault(hook_name, []).append(hook_func)

            logger.info(f"Loaded plugin: {plugin_name} v{plugin_class.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            self._plugins[Path(plugin_path).stem] = PluginInfo(
                name=Path(plugin_path).stem,
                version="?",
                description="",
                path=plugin_path,
                error=str(e)
            )
            return False

    async def unload(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        info = self._plugins.get(plugin_name)
        if not info or not info.instance:
            return False

        try:
            await info.instance.shutdown()
        except Exception as e:
            logger.warning(f"Plugin {plugin_name} shutdown error: {e}")

        # Remove tools
        for tool_name in list(self._tools.keys()):
            if tool_name.startswith(f"{plugin_name}.") or (
                info.instance and tool_name in info.instance.get_tools()
            ):
                self._tools.pop(tool_name, None)

        # Remove hooks
        if info.instance:
            for hook_name, hook_func in info.instance.get_hooks().items():
                if hook_name in self._hooks:
                    self._hooks[hook_name] = [
                        h for h in self._hooks[hook_name] if h != hook_func
                    ]

        del self._plugins[plugin_name]
        logger.info(f"Unloaded plugin: {plugin_name}")
        return True

    async def reload(self, plugin_name: str) -> bool:
        """Reload a plugin."""
        info = self._plugins.get(plugin_name)
        if not info:
            return False
        await self.unload(plugin_name)
        return await self.load(info.path)

    async def load_all(self) -> Dict[str, bool]:
        """Discover and load all plugins."""
        results = {}
        discovered = await self.discover()
        for path in discovered:
            name = Path(path).stem
            results[name] = await self.load(path)
        return results

    async def emit_event(self, event: str, data: Dict[str, Any] = None):
        """Send event to all enabled plugins."""
        data = data or {}
        for name, info in self._plugins.items():
            if info.enabled and info.instance:
                try:
                    await info.instance.on_event(event, data)
                except Exception as e:
                    logger.warning(f"Plugin {name} event {event} error: {e}")

    def get_tools(self) -> Dict[str, Callable]:
        """Get all registered tools from plugins."""
        return dict(self._tools)

    def get_hooks(self, hook_name: str) -> List[Callable]:
        """Get all hooks for a given lifecycle event."""
        return list(self._hooks.get(hook_name, []))

    def list_plugins(self) -> List[Dict]:
        """List all loaded plugins."""
        return [{
            "name": info.name,
            "version": info.version,
            "description": info.description,
            "enabled": info.enabled,
            "error": info.error,
            "tools": [t for t in self._tools if t.startswith(f"{info.name}.")],
        } for info in self._plugins.values()]

    def enable(self, plugin_name: str) -> bool:
        info = self._plugins.get(plugin_name)
        if info:
            info.enabled = True
            return True
        return False

    def disable(self, plugin_name: str) -> bool:
        info = self._plugins.get(plugin_name)
        if info:
            info.enabled = False
            return True
        return False
