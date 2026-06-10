"""
Code Intelligence — AST analysis, dependency graph, refactoring detection.
Agent can understand code structure, find patterns, suggest improvements.
"""
import os
import re
import ast
import logging
import hashlib
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Information about a function/method."""
    name: str
    file_path: str
    line: int
    end_line: int
    args: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    docstring: str = ""
    complexity: int = 0  # cyclomatic complexity estimate
    lines_of_code: int = 0
    calls: List[str] = field(default_factory=list)  # functions this calls
    called_by: List[str] = field(default_factory=list)
    returns: str = ""


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    file_path: str
    line: int
    end_line: int
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    docstring: str = ""
    lines_of_code: int = 0


@dataclass
class ImportInfo:
    """Information about an import."""
    module: str
    names: List[str] = field(default_factory=list)
    alias: str = ""
    file_path: str = ""
    line: int = 0
    is_from: bool = False


@dataclass
class CodeIssue:
    """A detected code issue."""
    severity: str  # error, warning, info
    category: str  # complexity, duplication, naming, security, style
    message: str
    file_path: str = ""
    line: int = 0
    suggestion: str = ""


@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    path: str
    language: str = "python"
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    issues: List[CodeIssue] = field(default_factory=list)
    complexity: int = 0  # file-level complexity


class CodeAnalyzer:
    """
    Python code analyzer with:
    - AST-based function/class extraction
    - Import dependency mapping
    - Cyclomatic complexity estimation
    - Code duplication detection
    - Naming convention checks
    - Dead code detection (unused imports/functions)
    - Dependency graph generation
    - Code metrics (LOC, comment ratio, etc.)
    """

    def __init__(self):
        self._files: Dict[str, FileAnalysis] = {}
        self._function_index: Dict[str, FunctionInfo] = {}
        self._class_index: Dict[str, ClassInfo] = {}
        self._import_graph: Dict[str, Set[str]] = defaultdict(set)
        self._call_graph: Dict[str, Set[str]] = defaultdict(set)

    def analyze_file(self, path: str) -> FileAnalysis:
        """Analyze a single Python file."""
        p = Path(path)
        if not p.exists() or p.suffix != '.py':
            return FileAnalysis(path=path)

        try:
            source = p.read_text(encoding='utf-8', errors='replace')
            tree = ast.parse(source, filename=path)
        except SyntaxError as e:
            analysis = FileAnalysis(path=path)
            analysis.issues.append(CodeIssue(
                severity="error", category="syntax",
                message="Syntax error: %s" % e, file_path=path, line=e.lineno or 0))
            return analysis

        analysis = FileAnalysis(path=path)
        lines = source.split('\n')
        analysis.total_lines = len(lines)
        analysis.blank_lines = sum(1 for l in lines if not l.strip())
        analysis.comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
        analysis.code_lines = analysis.total_lines - analysis.blank_lines - analysis.comment_lines

        # Extract functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._extract_function(node, path)
                analysis.functions.append(func)
                self._function_index["%s:%s" % (path, func.name)] = func

            elif isinstance(node, ast.ClassDef):
                cls = self._extract_class(node, path)
                analysis.classes.append(cls)
                self._class_index["%s:%s" % (path, cls.name)] = cls

        # Extract imports
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imp = ImportInfo(
                        module=alias.name, alias=alias.asname or "",
                        file_path=path, line=node.lineno)
                    analysis.imports.append(imp)
                    self._import_graph[path].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names]
                imp = ImportInfo(
                    module=node.module or "", names=names,
                    file_path=path, line=node.lineno, is_from=True)
                analysis.imports.append(imp)
                self._import_graph[path].add(node.module or "")

        # Build call graph
        for func in analysis.functions:
            for call_name in func.calls:
                self._call_graph[func.name].add(call_name)

        # Detect issues
        analysis.issues = self._detect_issues(analysis)
        analysis.complexity = sum(f.complexity for f in analysis.functions)

        self._files[path] = analysis
        return analysis

    def analyze_directory(self, path: str, recursive: bool = True) -> Dict[str, FileAnalysis]:
        """Analyze all Python files in a directory."""
        p = Path(path)
        results = {}
        pattern = "**/*.py" if recursive else "*.py"
        for f in p.glob(pattern):
            try:
                results[str(f)] = self.analyze_file(str(f))
            except Exception as e:
                logger.warning("Failed to analyze %s: %s" % (f, e))
        return results

    def _extract_function(self, node, path: str) -> FunctionInfo:
        """Extract function info from AST node."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(ast.dump(dec))

        # Extract calls
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)

        # Estimate complexity
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                  ast.With, ast.Assert, ast.BoolOp)):
                complexity += 1

        docstring = ast.get_docstring(node) or ""

        return FunctionInfo(
            name=node.name, file_path=path,
            line=node.lineno, end_line=node.end_lineno or node.lineno,
            args=args, decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            docstring=docstring, complexity=complexity,
            lines_of_code=(node.end_lineno or node.lineno) - node.lineno + 1,
            calls=calls,
        )

    def _extract_class(self, node, path: str) -> ClassInfo:
        """Extract class info from AST node."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append("%s.%s" % (ast.dump(base.value), base.attr))

        methods = []
        attributes = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(child.name)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)

        return ClassInfo(
            name=node.name, file_path=path,
            line=node.lineno, end_line=node.end_lineno or node.lineno,
            bases=bases, methods=methods, attributes=attributes,
            docstring=ast.get_docstring(node) or "",
            lines_of_code=(node.end_lineno or node.lineno) - node.lineno + 1,
        )

    def _detect_issues(self, analysis: FileAnalysis) -> List[CodeIssue]:
        """Detect code issues in a file."""
        issues = []

        # High complexity functions
        for func in analysis.functions:
            if func.complexity > 10:
                issues.append(CodeIssue(
                    severity="warning", category="complexity",
                    message="Function '%s' has high complexity (%d)" % (func.name, func.complexity),
                    file_path=analysis.path, line=func.line,
                    suggestion="Consider breaking into smaller functions"))

        # Long functions
        for func in analysis.functions:
            if func.lines_of_code > 100:
                issues.append(CodeIssue(
                    severity="warning", category="complexity",
                    message="Function '%s' is %d lines long" % (func.name, func.lines_of_code),
                    file_path=analysis.path, line=func.line,
                    suggestion="Consider extracting sub-functions"))

        # Missing docstrings
        for func in analysis.functions:
            if not func.docstring and not func.name.startswith('_'):
                issues.append(CodeIssue(
                    severity="info", category="style",
                    message="Public function '%s' missing docstring" % func.name,
                    file_path=analysis.path, line=func.line))

        # Naming conventions
        for func in analysis.functions:
            if func.name.startswith('_') and func.name != '__init__' and not func.name.startswith('__'):
                pass  # Private, OK
            elif not re.match(r'^[a-z][a-z0-9_]*$', func.name) and not func.name.startswith('__'):
                issues.append(CodeIssue(
                    severity="info", category="naming",
                    message="Function '%s' doesn't follow snake_case" % func.name,
                    file_path=analysis.path, line=func.line))

        return issues

    def get_dependency_graph(self) -> Dict[str, Set[str]]:
        """Get file-level import dependency graph."""
        return dict(self._import_graph)

    def get_call_graph(self, function_name: str = None) -> Dict[str, Set[str]]:
        """Get function call graph."""
        if function_name:
            return {function_name: self._call_graph.get(function_name, set())}
        return dict(self._call_graph)

    def find_function(self, name: str) -> List[FunctionInfo]:
        """Find functions by name across all analyzed files."""
        results = []
        for key, func in self._function_index.items():
            if func.name == name:
                results.append(func)
        return results

    def find_class(self, name: str) -> List[ClassInfo]:
        """Find classes by name."""
        results = []
        for key, cls in self._class_index.items():
            if cls.name == name:
                results.append(cls)
        return results

    def get_unused_imports(self, path: str) -> List[ImportInfo]:
        """Detect potentially unused imports."""
        analysis = self._files.get(path)
        if not analysis:
            return []

        # Get all names used in the file
        try:
            source = Path(path).read_text()
            tree = ast.parse(source)
        except Exception:
            return []

        used_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                used_names.add(node.attr)

        unused = []
        for imp in analysis.imports:
            if imp.is_from:
                for name in imp.names:
                    if name not in used_names and name != '*':
                        unused.append(imp)
            elif imp.alias:
                if imp.alias not in used_names:
                    unused.append(imp)
            else:
                top_module = imp.module.split('.')[0]
                if top_module not in used_names:
                    unused.append(imp)

        return unused

    def summary(self, path: str = None) -> Dict:
        """Get analysis summary."""
        if path:
            a = self._files.get(path)
            if not a:
                return {}
            return {
                "path": a.path,
                "lines": a.total_lines,
                "code_lines": a.code_lines,
                "functions": len(a.functions),
                "classes": len(a.classes),
                "imports": len(a.imports),
                "complexity": a.complexity,
                "issues": len(a.issues),
            }

        total_files = len(self._files)
        total_lines = sum(a.total_lines for a in self._files.values())
        total_functions = sum(len(a.functions) for a in self._files.values())
        total_classes = sum(len(a.classes) for a in self._files.values())
        total_issues = sum(len(a.issues) for a in self._files.values())
        return {
            "files": total_files,
            "total_lines": total_lines,
            "functions": total_functions,
            "classes": total_classes,
            "issues": total_issues,
            "complexity": sum(a.complexity for a in self._files.values()),
        }
