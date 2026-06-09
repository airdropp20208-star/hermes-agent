"""
Async Conversation Loop — lightweight async wrapper.

Wraps the existing sync conversation loop with async/await,
enabling continuous operation and concurrent tool execution.

The heavy logic stays in conversation_loop.py (unchanged).
This module adds the async layer on top.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


class AsyncConversationLoop:
    """
    Async conversation loop that wraps the sync core.
    
    Two modes:
      1. Single-turn:  response = await loop.run("hello")
      2. Continuous:   await loop.run_forever(handler)
    """
    
    def __init__(self, agent, max_concurrent_tools: int = 5):
        self.agent = agent
        self.max_concurrent_tools = max_concurrent_tools
        self._running = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
        
    async def run(self, user_message: str, **kwargs) -> Dict[str, Any]:
        """Run one conversation turn asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.agent.run_conversation(user_message, **kwargs)
        )
    
    async def run_batch(self, messages: List[str]) -> List[Dict[str, Any]]:
        """Run multiple messages concurrently."""
        tasks = [self.run(msg) for msg in messages]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def run_forever(self, 
                          message_source: Callable[[], Awaitable[Optional[str]]],
                          response_sink: Callable[[Dict], Awaitable[None]],
                          poll_interval: float = 0.5):
        """
        Run forever — process messages as they arrive.
        
        Args:
            message_source: async callable that returns next message or None
            response_sink: async callable that receives each result
            poll_interval: seconds to wait when idle
        """
        self._running = True
        logger.info("Continuous loop started")
        
        while self._running:
            try:
                msg = await message_source()
                if msg:
                    result = await self.run(msg)
                    await response_sink(result)
                else:
                    await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(poll_interval)
                
        logger.info("Continuous loop stopped")
    
    def stop(self):
        self._running = False


class AsyncToolRunner:
    """
    Run multiple tool calls concurrently instead of sequentially.
    
    The sync loop does:
        for tc in tool_calls:
            result = handle_function_call(tc.name, tc.args)
    
    This does:
        results = await asyncio.gather(*[handle(tc) for tc in tool_calls])
    """
    
    def __init__(self, dispatcher: Callable, max_concurrent: int = 5):
        self.dispatcher = dispatcher
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def run_batch(self, tool_calls: list) -> list:
        """Execute tool calls concurrently."""
        async def _run_one(tc):
            async with self.semaphore:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.dispatcher(tc["name"], tc["args"])
                    )
                    return {"id": tc["id"], "result": result}
                except Exception as e:
                    return {"id": tc["id"], "error": str(e)}
                    
        return await asyncio.gather(*[_run_one(tc) for tc in tool_calls])


# ============================================================================
# Integration helper — patch the existing sync loop to support async
# ============================================================================

def make_run_conversation_async(conversation_loop_module):
    """
    Monkey-patch the sync run_conversation to also work with await.
    
    Usage:
        from agent.async_core import make_run_conversation_async
        from agent import conversation_loop
        make_run_conversation_async(conversation_loop)
        
        # Now both work:
        result = conversation_loop.run_conversation(agent, msg)  # sync
        result = await conversation_loop.run_conversation_async(agent, msg)  # async
    """
    original = conversation_loop_module.run_conversation
    
    async def run_conversation_async(agent, user_message, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: original(agent, user_message, **kwargs)
        )
    
    conversation_loop_module.run_conversation_async = run_conversation_async
    logger.info("Patched conversation_loop with async support")
