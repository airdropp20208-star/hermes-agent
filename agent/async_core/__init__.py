"""
Async Conversation Loop for Hermes Agent

Converts the sync conversation loop to async, enabling:
- Continuous operation (agent stays on, processes messages)
- Non-blocking API calls
- Concurrent tool execution
- Better resource utilization

Usage:
    from agent.async_core.async_loop import AsyncConversationLoop
    loop = AsyncConversationLoop(agent)
    result = await loop.run("Hello!")
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class AsyncConversationLoop:
    """
    Async wrapper around the conversation loop.
    
    Supports two modes:
    1. Single-turn: await loop.run(message) → response
    2. Continuous: await loop.start_continuous() → runs forever
    """
    
    def __init__(self, agent, max_concurrent_tools: int = 5):
        self.agent = agent
        self.max_concurrent_tools = max_concurrent_tools
        self._running = False
        self._message_queue = asyncio.Queue()
        self._response_futures: Dict[str, asyncio.Future] = {}
        
    async def run(self, user_message: str, **kwargs) -> Dict[str, Any]:
        """
        Run a single conversation turn asynchronously.
        
        Args:
            user_message: The user's message
            **kwargs: Additional args passed to run_conversation
            
        Returns:
            Dict with final_response and messages
        """
        # Delegate to the sync loop in a thread pool
        # This keeps the existing sync code working while we go async
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            lambda: self.agent.run_conversation(user_message, **kwargs)
        )
        
        return result
    
    async def run_with_tools(self, user_message: str, 
                              tool_executor: Optional[Callable] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Run conversation with concurrent tool execution.
        
        When the LLM requests multiple tool calls, they can execute
        in parallel instead of sequentially.
        """
        result = await self.run(user_message, **kwargs)
        
        # If there are pending tool calls, execute them concurrently
        if tool_executor and result.get("pending_tool_calls"):
            tasks = [
                tool_executor(tc["name"], tc["args"]) 
                for tc in result["pending_tool_calls"]
            ]
            tool_results = await asyncio.gather(*tasks, return_exceptions=True)
            result["tool_results"] = tool_results
            
        return result
    
    async def start_continuous(self, 
                                message_handler: Optional[Callable] = None,
                                poll_interval: float = 1.0,
                                on_idle: Optional[Callable] = None):
        """
        Start continuous operation mode.
        
        The agent stays on and processes messages as they arrive.
        Can be used with gateway, webhooks, or any message source.
        
        Args:
            message_handler: Async function that yields messages
                           Signature: async def handler() -> (message, response_callback)
            poll_interval: Seconds between idle polls
            on_idle: Called when no messages to process
        """
        self._running = True
        logger.info("Starting continuous conversation loop")
        
        while self._running:
            try:
                if message_handler:
                    # Get next message from handler
                    msg = await message_handler()
                    if msg:
                        message, callback = msg
                        result = await self.run(message)
                        if callback:
                            await callback(result)
                    else:
                        if on_idle:
                            await on_idle()
                        await asyncio.sleep(poll_interval)
                else:
                    # Queue-based mode
                    try:
                        message, future = await asyncio.wait_for(
                            self._message_queue.get(), 
                            timeout=poll_interval
                        )
                        result = await self.run(message)
                        future.set_result(result)
                    except asyncio.TimeoutError:
                        if on_idle:
                            await on_idle()
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous loop: {e}")
                await asyncio.sleep(poll_interval)
                
        logger.info("Continuous conversation loop stopped")
    
    def stop(self):
        """Stop the continuous loop."""
        self._running = False
    
    async def send_message(self, message: str) -> Dict[str, Any]:
        """
        Send a message to the running continuous agent.
        Returns a future that resolves with the response.
        """
        future = asyncio.get_event_loop().create_future()
        await self._message_queue.put((message, future))
        return await future


class AsyncToolExecutor:
    """
    Execute multiple tool calls concurrently.
    
    The sync version processes tools one-by-one.
    This version can run multiple tools in parallel.
    """
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
    async def execute_batch(self, tool_calls: List[Dict], 
                            dispatcher: Callable) -> List[Dict]:
        """
        Execute a batch of tool calls concurrently.
        
        Args:
            tool_calls: List of {name, args, id} dicts
            dispatcher: Async function to dispatch individual tool calls
            
        Returns:
            List of tool results
        """
        async def _execute_one(tc):
            async with self._semaphore:
                try:
                    result = await dispatcher(tc["name"], tc["args"])
                    return {"tool_call_id": tc["id"], "result": result}
                except Exception as e:
                    return {"tool_call_id": tc["id"], "error": str(e)}
                    
        tasks = [_execute_one(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks)
        return list(results)


class ContinuousGateway:
    """
    Always-on gateway that processes messages continuously.
    
    Integrates with the existing gateway platform adapters
    but runs in continuous async mode.
    """
    
    def __init__(self, agent_factory: Callable, platforms: List[str] = None):
        """
        Args:
            agent_factory: Callable that creates AIAgent instances
            platforms: List of platform names to run
        """
        self.agent_factory = agent_factory
        self.platforms = platforms or []
        self._running = False
        self._agents: Dict[str, Any] = {}
        
    async def start(self):
        """Start the continuous gateway."""
        self._running = True
        logger.info(f"Starting continuous gateway for platforms: {self.platforms}")
        
        # Create agent for each platform
        for platform in self.platforms:
            self._agents[platform] = self.agent_factory(platform=platform)
            
        # Keep running
        while self._running:
            await asyncio.sleep(1)
            
    def stop(self):
        """Stop the gateway."""
        self._running = False
