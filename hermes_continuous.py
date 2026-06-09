#!/usr/bin/env python3
"""
Hermes Agent - Continuous Mode

Run Hermes as an always-on agent that processes messages continuously.
Supports multiple input sources:
- Telegram/Discord/Slack (via gateway)
- Webhook API
- CLI interactive mode

Usage:
    python -m hermes_continuous --mode gateway --platforms telegram,discord
    python -m hermes_continuous --mode webhook --port 8080
    python -m hermes_continuous --mode cli
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.async_core import AsyncConversationLoop, ContinuousGateway


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_agent(platform: str = None):
    """Create an AIAgent instance."""
    from run_agent import AIAgent
    return AIAgent(platform=platform)


async def run_gateway_mode(platforms: list, config: dict):
    """Run as continuous gateway."""
    gateway = ContinuousGateway(
        agent_factory=lambda platform=None: create_agent(platform),
        platforms=platforms
    )
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, gateway.stop)
    
    await gateway.start()


async def run_webhook_mode(port: int, config: dict):
    """Run as webhook API server."""
    try:
        from aiohttp import web
    except ImportError:
        logger.error("aiohttp required for webhook mode: pip install aiohttp")
        return
    
    agent = create_agent()
    async_loop = AsyncConversationLoop(agent)
    
    async def handle_message(request):
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return web.json_response({"error": "No message provided"}, status=400)
        
        result = await async_loop.run(message)
        return web.json_response({
            "response": result.get("final_response", ""),
            "status": "ok"
        })
    
    app = web.Application()
    app.router.add_post("/chat", handle_message)
    app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    
    logger.info(f"Webhook server running on port {port}")
    
    # Keep running
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


async def run_cli_mode(config: dict):
    """Run in interactive CLI mode."""
    agent = create_agent(platform="cli")
    async_loop = AsyncConversationLoop(agent)
    
    print("Hermes Continuous Mode (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        try:
            message = input("\nYou: ").strip()
            if message.lower() in ("quit", "exit", "q"):
                break
                
            if not message:
                continue
                
            result = await async_loop.run(message)
            response = result.get("final_response", "No response")
            print(f"\nHermes: {response}")
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Hermes Agent - Continuous Mode")
    parser.add_argument("--mode", choices=["gateway", "webhook", "cli"], 
                       default="cli", help="Run mode")
    parser.add_argument("--platforms", type=str, default="",
                       help="Comma-separated platform list (gateway mode)")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for webhook mode")
    parser.add_argument("--config", type=str, default="",
                       help="Config file path")
    
    args = parser.parse_args()
    
    config = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    
    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    
    # Run the appropriate mode
    if args.mode == "gateway":
        asyncio.run(run_gateway_mode(platforms, config))
    elif args.mode == "webhook":
        asyncio.run(run_webhook_mode(args.port, config))
    elif args.mode == "cli":
        asyncio.run(run_cli_mode(config))


if __name__ == "__main__":
    main()
