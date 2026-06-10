"""
Dependency Injection Container — wire up all modules.
Provides centralized configuration and lifecycle management.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, List, Type, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Container:
    """
    DI Container with:
    - Singleton and factory registrations
    - Lazy initialization
    - Lifecycle management (init/shutdown)
    - Dependency resolution
    - Auto-wiring
    """

    def __init__(self):
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._initialized: List[str] = []
        self._shutdown_hooks: List[Callable] = []

    def register_singleton(self, name: str, instance: Any):
        """Register a pre-created singleton."""
        self._singletons[name] = instance

    def register_factory(self, name: str, factory: Callable):
        """Register a factory (called on first get)."""
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """Get a registered service."""
        if name in self._singletons:
            return self._singletons[name]
        if name in self._factories:
            instance = self._factories[name]()
            self._singletons[name] = instance
            del self._factories[name]
            return instance
        raise KeyError(f"Service '{name}' not registered")

    def has(self, name: str) -> bool:
        return name in self._singletons or name in self._factories

    async def initialize(self):
        """Initialize all registered services."""
        for name, instance in self._singletons.items():
            if hasattr(instance, 'initialize') and name not in self._initialized:
                try:
                    if asyncio.iscoroutinefunction(instance.initialize):
                        await instance.initialize()
                    else:
                        instance.initialize()
                    self._initialized.append(name)
                    logger.info(f"Initialized: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize {name}: {e}")

    async def shutdown(self):
        """Shutdown all services in reverse order."""
        for hook in reversed(self._shutdown_hooks):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                logger.warning(f"Shutdown hook error: {e}")

        for name in reversed(self._initialized):
            instance = self._singletons.get(name)
            if instance and hasattr(instance, 'shutdown'):
                try:
                    if asyncio.iscoroutinefunction(instance.shutdown):
                        await instance.shutdown()
                    else:
                        instance.shutdown()
                    logger.info(f"Shutdown: {name}")
                except Exception as e:
                    logger.warning(f"Shutdown {name} error: {e}")

    def on_shutdown(self, hook: Callable):
        """Register a shutdown hook."""
        self._shutdown_hooks.append(hook)

    def list_services(self) -> Dict[str, str]:
        """List all registered services."""
        services = {}
        for name in self._singletons:
            services[name] = type(self._singletons[name]).__name__
        for name in self._factories:
            services[name] = f"factory:{self._factories[name].__name__}"
        return services


def create_default_container(config: Dict = None) -> Container:
    """Create a container with all default async core services."""
    from .config import ConfigManager
    from .events import EventBus
    from .memory import MemoryStore
    from .sessions import SessionManager
    from .budget import BudgetTracker
    from .streaming import StreamingEngine
    from .health import HealthMonitor
    from .plugins import PluginEngine

    config = config or {}
    container = Container()

    # Config
    cfg = ConfigManager(config.get("config_dir"), config.get("profile", "default"))
    container.register_singleton("config", cfg)

    # Events
    container.register_singleton("events", EventBus())

    # Memory
    mem_path = cfg.get("memory.db_path", "").replace("~", str(__import__('os').path.expanduser("~")))
    container.register_singleton("memory", MemoryStore(db_path=mem_path))

    # Sessions
    sess_path = cfg.get("sessions.db_path", "").replace("~", str(__import__('os').path.expanduser("~")))
    container.register_singleton("sessions", SessionManager(db_path=sess_path))

    # Budget
    container.register_singleton("budget", BudgetTracker())

    # Streaming
    container.register_singleton("streaming", StreamingEngine())

    # Health
    container.register_singleton("health", HealthMonitor())

    # Plugins
    dirs = cfg.get("plugins.directories", [])
    container.register_singleton("plugins", PluginEngine(plugin_dirs=dirs))

    return container
