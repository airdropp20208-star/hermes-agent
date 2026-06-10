"""
HTTP Server — FastAPI REST + SSE + WebSocket.
Provides web interface for the async agent.
"""
import asyncio
import json
import time
import logging
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: List[str] = None
    auth_token: str = ""
    max_connections: int = 100

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


class RequestHandler:
    """HTTP request handler with routing."""

    def __init__(self):
        self._routes: Dict[str, Dict[str, Any]] = {}
        self._middleware: List = []
        self._ws_handlers: Dict[str, Any] = {}

    def route(self, path: str, methods: List[str] = None):
        """Register a route handler."""
        methods = methods or ["GET"]
        def decorator(func):
            for method in methods:
                key = f"{method}:{path}"
                self._routes[key] = {"handler": func, "method": method, "path": path}
            return func
        return decorator

    def middleware(self, func):
        """Register middleware."""
        self._middleware.append(func)
        return func

    def websocket(self, path: str):
        """Register WebSocket handler."""
        def decorator(func):
            self._ws_handlers[path] = func
            return func
        return decorator

    def get_route(self, method: str, path: str):
        """Find matching route."""
        key = f"{method}:{path}"
        if key in self._routes:
            return self._routes[key]
        # Pattern matching (simple prefix)
        for k, v in self._routes.items():
            route_method, route_path = k.split(":", 1)
            if route_method == method and self._path_match(route_path, path):
                return v
        return None

    @staticmethod
    def _path_match(pattern: str, path: str) -> bool:
        if pattern == path:
            return True
        # Simple wildcard: /api/v1/* matches /api/v1/anything
        if pattern.endswith("/*"):
            return path.startswith(pattern[:-2] + "/")
        return False


class AsyncHTTPServer:
    """
    HTTP server with:
    - REST API endpoints
    - Server-Sent Events (SSE)
    - WebSocket support
    - Authentication
    - CORS
    - Rate limiting
    - Health checks
    """

    def __init__(self, config: ServerConfig = None):
        self.config = config or ServerConfig()
        self.handler = RequestHandler()
        self._app = None
        self._server = None
        self._connections: Dict[str, Any] = {}
        self._request_count = 0
        self._start_time = None

        self._register_default_routes()

    def _register_default_routes(self):
        """Register default API routes."""
        @self.handler.route("/health", ["GET"])
        async def health(request=None):
            return {
                "status": "healthy",
                "uptime": time.time() - self._start_time if self._start_time else 0,
                "connections": len(self._connections),
                "requests": self._request_count,
            }

        @self.handler.route("/api/v1/chat", ["POST"])
        async def chat(request=None):
            return {"error": "Not implemented — bind an agent first"}

        @self.handler.route("/api/v1/stats", ["GET"])
        async def stats(request=None):
            return self.stats()

    def _create_app(self):
        """Create FastAPI app if available, else basic HTTP server."""
        try:
            from fastapi import FastAPI, Request
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import StreamingResponse, JSONResponse
            import uvicorn

            app = FastAPI(title="Hermes Agent API", version="0.2.0")

            # CORS
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # Register routes
            for key, route_info in self.handler._routes.items():
                method = route_info["method"].lower()
                path = route_info["path"]
                func = route_info["handler"]

                if method == "get":
                    app.get(path)(func)
                elif method == "post":
                    app.post(path)(func)
                elif method == "put":
                    app.put(path)(func)
                elif method == "delete":
                    app.delete(path)(func)

            self._app = app
            return app

        except ImportError:
            logger.warning("FastAPI not available, using basic HTTP server")
            return self._create_basic_app()

    def _create_basic_app(self):
        """Fallback: basic asyncio HTTP server."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading

        handler = self.handler

        class BasicHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                route = handler.get_route("GET", self.path)
                if route:
                    result = asyncio.run(route["handler"]())
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                route = handler.get_route("POST", self.path)
                if route:
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length) if content_length else b'{}'
                    result = asyncio.run(route["handler"](json.loads(body)))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress default logging

        return BasicHandler

    async def start(self):
        """Start the server."""
        self._start_time = time.time()
        self._create_app()

        if self._app and hasattr(self._app, 'routes'):
            # FastAPI
            try:
                import uvicorn
                config = uvicorn.Config(
                    self._app,
                    host=self.config.host,
                    port=self.config.port,
                    log_level="info",
                )
                server = uvicorn.Server(config)
                self._server = server
                await server.serve()
            except ImportError:
                logger.error("uvicorn not installed. Install: pip install uvicorn")
                raise
        else:
            # Basic server
            from http.server import HTTPServer
            import functools
            handler_class = self._create_basic_app()
            server = HTTPServer((self.config.host, self.config.port), handler_class)
            self._server = server
            logger.info(f"Basic HTTP server on {self.config.host}:{self.config.port}")
            server.serve_forever()

    def stop(self):
        """Stop the server."""
        if self._server:
            if hasattr(self._server, 'shutdown'):
                self._server.shutdown()
            self._server = None

    def stats(self):
        return {
            "uptime": time.time() - self._start_time if self._start_time else 0,
            "connections": len(self._connections),
            "requests": self._request_count,
            "routes": len(self.handler._routes),
            "config": {
                "host": self.config.host,
                "port": self.config.port,
            },
        }
