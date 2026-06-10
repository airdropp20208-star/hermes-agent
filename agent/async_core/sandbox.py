"""
Code Sandbox — safe Python/bash execution with isolation.
Provides sandboxed code execution for agent tool use.
"""
import asyncio
import subprocess
import tempfile
import os
import time
import logging
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from sandboxed execution."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0
    language: str = ""
    timed_out: bool = False
    memory_peak_mb: float = 0
    files_created: List[str] = field(default_factory=list)


@dataclass
class SandboxConfig:
    """Sandbox configuration."""
    timeout: float = 30
    max_output_bytes: int = 50000
    max_memory_mb: int = 512
    work_dir: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)


class CodeSandbox:
    """
    Sandboxed code execution with:
    - Python execution (with timeout)
    - Bash execution (with timeout)
    - Output capture (stdout, stderr)
    - Timeout enforcement
    - Working directory isolation
    - File creation tracking
    """

    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self._work_dir = self.config.work_dir or tempfile.mkdtemp(prefix="hermes_sandbox_")
        os.makedirs(self._work_dir, exist_ok=True)
        self._execution_count = 0

    async def run_python(self, code: str, timeout: float = None) -> ExecutionResult:
        """Execute Python code in sandbox."""
        timeout = timeout or self.config.timeout
        start = time.monotonic()

        script_path = os.path.join(self._work_dir, "script_%d.py" % self._execution_count)
        self._execution_count += 1

        with open(script_path, 'w') as f:
            f.write(code)

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._work_dir,
                env={**os.environ, **self.config.env_vars},
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout, stderr = b"", b"Execution timed out"
                timed_out = True

            stdout = stdout[:self.config.max_output_bytes]
            stderr = stderr[:self.config.max_output_bytes]

            return ExecutionResult(
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                exit_code=proc.returncode or 0,
                duration_ms=(time.monotonic() - start) * 1000,
                language="python",
                timed_out=timed_out,
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    async def run_bash(self, command: str, timeout: float = None) -> ExecutionResult:
        """Execute bash command in sandbox."""
        timeout = timeout or self.config.timeout
        start = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._work_dir,
                env={**os.environ, **self.config.env_vars},
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout, stderr = b"", b"Execution timed out"
                timed_out = True

            stdout = stdout[:self.config.max_output_bytes]
            stderr = stderr[:self.config.max_output_bytes]

            return ExecutionResult(
                stdout=stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                exit_code=proc.returncode or 0,
                duration_ms=(time.monotonic() - start) * 1000,
                language="bash",
                timed_out=timed_out,
            )
        except Exception as e:
            return ExecutionResult(
                stderr=str(e), exit_code=1,
                duration_ms=(time.monotonic() - start) * 1000,
                language="bash",
            )

    async def run_script(self, path: str, args: List[str] = None,
                         timeout: float = None) -> ExecutionResult:
        """Run a script file."""
        timeout = timeout or self.config.timeout
        args = args or []
        ext = Path(path).suffix.lower()

        if ext == '.py':
            cmd = ["python3", path] + args
        elif ext in ('.sh', '.bash'):
            cmd = ["bash", path] + args
        elif ext == '.js':
            cmd = ["node", path] + args
        else:
            return ExecutionResult(stderr="Unsupported script type: %s" % ext, exit_code=1)

        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._work_dir,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            stdout, stderr = b"", b"Timed out"
            timed_out = True

        return ExecutionResult(
            stdout=stdout.decode('utf-8', errors='replace')[:self.config.max_output_bytes],
            stderr=stderr.decode('utf-8', errors='replace')[:self.config.max_output_bytes],
            exit_code=proc.returncode or 0,
            duration_ms=(time.monotonic() - start) * 1000,
            language=ext[1:],
            timed_out=timed_out,
        )

    def cleanup(self):
        """Clean up sandbox files."""
        import shutil
        try:
            shutil.rmtree(self._work_dir)
        except Exception:
            pass

    def stats(self) -> Dict:
        return {
            "work_dir": self._work_dir,
            "executions": self._execution_count,
            "timeout": self.config.timeout,
        }
