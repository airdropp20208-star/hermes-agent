"""
VPS Skill Pipeline Tool

Automates skill creation and testing on VPS:
1. Agent creates skill on VPS
2. Auto-tests on VPS
3. Pushes to GitHub
4. User installs locally

Usage:
    From agent: vps_skill_create(name="my-skill", content="...")
    From agent: vps_skill_test(name="my-skill")
    From agent: vps_skill_push(name="my-skill")
"""

import json
import os
import subprocess
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path

from tools.registry import registry


def _get_hermes_home():
    """Get Hermes home directory."""
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def vps_skill_create(name: str, content: str, category: str = "custom",
                     description: str = "", triggers: str = "") -> str:
    """
    Create a new skill on VPS for testing.
    
    Args:
        name: Skill name (lowercase, hyphens)
        content: SKILL.md content
        category: Skill category
        description: Brief description
        triggers: When to use this skill
        
    Returns:
        JSON with status and path
    """
    try:
        # Validate name
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            return json.dumps({"error": "Invalid skill name. Use lowercase letters, hyphens, underscores."})
        
        # Build SKILL.md with frontmatter
        skill_md = f"""---
name: {name}
description: "{description}"
version: 0.1.0
author: agent
category: {category}
triggers:
  - {triggers}
---

# {name.replace('-', ' ').replace('_', ' ').title()}

{content}
"""
        
        # Create skill directory
        skill_dir = Path(tempfile.mkdtemp(prefix=f"skill_{name}_"))
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(skill_md)
        
        return json.dumps({
            "success": True,
            "name": name,
            "path": str(skill_dir),
            "skill_file": str(skill_file),
            "message": f"Skill '{name}' created at {skill_dir}"
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to create skill: {str(e)}"})


def vps_skill_test(name: str, skill_path: str, test_cases: Optional[str] = None) -> str:
    """
    Test a skill on VPS.
    
    Args:
        name: Skill name
        skill_path: Path to skill directory
        test_cases: Optional test scenarios (comma-separated)
        
    Returns:
        JSON with test results
    """
    try:
        skill_file = Path(skill_path) / "SKILL.md"
        if not skill_file.exists():
            return json.dumps({"error": f"Skill file not found: {skill_file}"})
        
        content = skill_file.read_text()
        
        # Basic validation
        results = {
            "has_frontmatter": content.startswith("---"),
            "has_name": "name:" in content[:200],
            "has_description": "description:" in content[:200],
            "has_content": len(content) > 100,
            "content_length": len(content),
        }
        
        # Check for common issues
        issues = []
        if not results["has_frontmatter"]:
            issues.append("Missing YAML frontmatter")
        if not results["has_name"]:
            issues.append("Missing 'name' in frontmatter")
        if not results["has_description"]:
            issues.append("Missing 'description' in frontmatter")
        if results["content_length"] < 50:
            issues.append("Content too short (< 50 chars)")
            
        results["issues"] = issues
        results["passed"] = len(issues) == 0
        
        return json.dumps({
            "success": True,
            "name": name,
            "results": results,
            "message": f"Tests {'PASSED' if results['passed'] else 'FAILED'}: {len(issues)} issues"
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to test skill: {str(e)}"})


def vps_skill_push(name: str, skill_path: str, 
                   repo_url: str = "https://github.com/airdropp20208-star/hermes-agent.git",
                   branch: str = "main") -> str:
    """
    Push skill to GitHub for user to install.
    
    Args:
        name: Skill name
        skill_path: Path to skill directory
        repo_url: GitHub repo URL
        branch: Branch to push to
        
    Returns:
        JSON with push status
    """
    try:
        skill_file = Path(skill_path) / "SKILL.md"
        if not skill_file.exists():
            return json.dumps({"error": f"Skill file not found: {skill_file}"})
        
        # Create skills directory in repo
        skills_dir = Path(skill_path).parent / "hermes-agent" / "skills" / "custom"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy skill
        import shutil
        target_dir = skills_dir / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(skill_path, target_dir)
        
        # Git operations
        repo_dir = Path(skill_path).parent / "hermes-agent"
        
        result = subprocess.run(
            ["git", "add", f"skills/custom/{name}"],
            capture_output=True, text=True, cwd=str(repo_dir)
        )
        
        result = subprocess.run(
            ["git", "commit", "-m", f"feat: add skill '{name}'"],
            capture_output=True, text=True, cwd=str(repo_dir)
        )
        
        result = subprocess.run(
            ["git", "push", "origin", branch],
            capture_output=True, text=True, cwd=str(repo_dir)
        )
        
        if result.returncode != 0:
            return json.dumps({"error": f"Push failed: {result.stderr[:500]}"})
        
        return json.dumps({
            "success": True,
            "name": name,
            "repo": repo_url,
            "branch": branch,
            "message": f"Skill '{name}' pushed to GitHub. User can install with: hermes skills install {name}"
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to push skill: {str(e)}"})


# Register tools
registry.register(
    name="vps_skill_create",
    toolset="terminal",
    schema={
        "name": "vps_skill_create",
        "description": "Create a new skill on VPS for testing. Returns path to created skill.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name (lowercase, hyphens)"},
                "content": {"type": "string", "description": "SKILL.md content"},
                "category": {"type": "string", "description": "Skill category", "default": "custom"},
                "description": {"type": "string", "description": "Brief description"},
                "triggers": {"type": "string", "description": "When to use this skill"}
            },
            "required": ["name", "content"]
        }
    },
    handler=lambda args, **kw: vps_skill_create(
        name=args.get("name", ""),
        content=args.get("content", ""),
        category=args.get("category", "custom"),
        description=args.get("description", ""),
        triggers=args.get("triggers", "")
    ),
)

registry.register(
    name="vps_skill_test",
    toolset="terminal",
    schema={
        "name": "vps_skill_test",
        "description": "Test a skill on VPS. Validates structure and content.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name"},
                "skill_path": {"type": "string", "description": "Path to skill directory"},
                "test_cases": {"type": "string", "description": "Test scenarios (comma-separated)"}
            },
            "required": ["name", "skill_path"]
        }
    },
    handler=lambda args, **kw: vps_skill_test(
        name=args.get("name", ""),
        skill_path=args.get("skill_path", ""),
        test_cases=args.get("test_cases")
    ),
)

registry.register(
    name="vps_skill_push",
    toolset="terminal",
    schema={
        "name": "vps_skill_push",
        "description": "Push skill to GitHub for user to install.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name"},
                "skill_path": {"type": "string", "description": "Path to skill directory"},
                "repo_url": {"type": "string", "description": "GitHub repo URL"},
                "branch": {"type": "string", "description": "Branch to push to", "default": "main"}
            },
            "required": ["name", "skill_path"]
        }
    },
    handler=lambda args, **kw: vps_skill_push(
        name=args.get("name", ""),
        skill_path=args.get("skill_path", ""),
        repo_url=args.get("repo_url", "https://github.com/airdropp20208-star/hermes-agent.git"),
        branch=args.get("branch", "main")
    ),
)
