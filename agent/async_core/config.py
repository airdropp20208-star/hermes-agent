"""
Config Manager — YAML, env vars, API keys, profiles.
Handles all configuration for the async core.
"""
import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "version": "0.2.0",
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "",
        "base_url": "",
        "max_tokens": 4096,
        "temperature": 0.7,
        "max_retries": 3,
        "timeout": 120,
        "fallback_provider": "",
        "fallback_model": "",
    },
    "memory": {
        "backend": "sqlite",
        "db_path": "~/.hermes/async_core/memory.db",
        "vector_dimension": 384,
        "max_entries": 10000,
        "auto_cleanup_days": 30,
    },
    "sessions": {
        "db_path": "~/.hermes/async_core/sessions.db",
        "max_sessions": 100,
        "auto_archive_days": 30,
    },
    "budget": {
        "enabled": True,
        "daily_limit_usd": 10.0,
        "monthly_limit_usd": 100.0,
        "warn_at_percent": 80,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8080,
        "cors_origins": ["*"],
        "auth_token": "",
    },
    "plugins": {
        "directories": [],
        "auto_load": True,
    },
    "orchestrator": {
        "max_concurrent_agents": 10,
        "max_depth": 3,
        "default_timeout": 300,
    },
    "logging": {
        "level": "INFO",
        "file": "~/.hermes/async_core/agent.log",
        "max_size_mb": 50,
        "backup_count": 5,
    },
}


class ConfigManager:
    """
    Configuration management with:
    - YAML file support
    - Environment variable override
    - Profile support (default, dev, prod)
    - API key management
    - Runtime config updates
    - Validation
    """

    def __init__(self, config_dir: str = None, profile: str = "default"):
        self.config_dir = Path(config_dir or os.path.expanduser("~/.hermes/async_core"))
        self.profile = profile
        self._config: Dict[str, Any] = {}
        self._env_prefix = "HERMES_"
        self._watchers: List = []

        self._load()

    def _config_path(self) -> Path:
        """Get config file path for current profile."""
        if self.profile == "default":
            return self.config_dir / "config.yaml"
        return self.config_dir / f"config.{self.profile}.yaml"

    def _load(self):
        """Load config from file and environment."""
        # Start with defaults
        self._config = self._deep_copy(DEFAULT_CONFIG)

        # Load from file
        path = self._config_path()
        if path.exists():
            try:
                import yaml
                with open(path) as f:
                    file_config = yaml.safe_load(f) or {}
                self._deep_merge(self._config, file_config)
            except ImportError:
                # Try JSON fallback
                json_path = path.with_suffix('.json')
                if json_path.exists():
                    with open(json_path) as f:
                        file_config = json.load(f)
                    self._deep_merge(self._config, file_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")

        # Override from environment
        self._load_env()

        # Create config dir
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _load_env(self):
        """Load config overrides from environment variables."""
        env_map = {
            f"{self._env_prefix}LLM_PROVIDER": ("llm", "provider"),
            f"{self._env_prefix}LLM_MODEL": ("llm", "model"),
            f"{self._env_prefix}LLM_API_KEY": ("llm", "api_key"),
            f"{self._env_prefix}LLM_BASE_URL": ("llm", "base_url"),
            f"{self._env_prefix}OPENAI_API_KEY": ("llm", "api_key"),
            f"{self._env_prefix}ANTHROPIC_API_KEY": ("llm", "api_key"),
            f"{self._env_prefix}SERVER_PORT": ("server", "port"),
            f"{self._env_prefix}SERVER_HOST": ("server", "host"),
            f"{self._env_prefix}LOG_LEVEL": ("logging", "level"),
        }
        for env_var, path in env_map.items():
            value = os.environ.get(env_var)
            if value:
                self._set_nested(path, value)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get config value by dotted key (e.g., 'llm.model')."""
        keys = dotted_key.split(".")
        obj = self._config
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]
            else:
                return default
        return obj

    def set(self, dotted_key: str, value: Any):
        """Set config value by dotted key."""
        keys = dotted_key.split(".")
        obj = self._config
        for k in keys[:-1]:
            if k not in obj:
                obj[k] = {}
            obj = obj[k]
        obj[keys[-1]] = value

        for watcher in self._watchers:
            try:
                watcher(dotted_key, value)
            except Exception:
                pass

    def get_section(self, section: str) -> Dict:
        """Get entire config section."""
        return dict(self._config.get(section, {}))

    def save(self):
        """Save current config to file."""
        path = self._config_path()
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        except ImportError:
            json_path = path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(self._config, f, indent=2)

    def get_api_key(self, provider: str = "") -> str:
        """Get API key for a provider."""
        key = self.get("llm.api_key", "")
        if not key:
            env_keys = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }
            env_var = env_keys.get(provider.lower(), "")
            if env_var:
                key = os.environ.get(env_var, "")
        return key

    def watch(self, callback):
        """Register config change watcher."""
        self._watchers.append(callback)

    def _set_nested(self, path: tuple, value):
        obj = self._config
        for k in path[:-1]:
            if k not in obj:
                obj[k] = {}
            obj = obj[k]
        obj[path[-1]] = value

    @staticmethod
    def _deep_merge(base: dict, override: dict):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                ConfigManager._deep_merge(base[k], v)
            else:
                base[k] = v

    @staticmethod
    def _deep_copy(d: dict) -> dict:
        import copy
        return copy.deepcopy(d)

    def __repr__(self):
        return f"ConfigManager(profile={self.profile}, dir={self.config_dir})"
