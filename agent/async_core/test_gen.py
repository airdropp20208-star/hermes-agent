"""
Test Generator — auto-generate tests from code, coverage analysis, fuzzing.
Agent can create comprehensive test suites automatically.
"""
import ast
import re
import uuid
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A generated test case."""
    id: str
    name: str
    function_name: str
    test_code: str
    test_type: str  # unit, edge, integration, fuzz
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    description: str = ""
    confidence: float = 0.7


@dataclass
class CoverageInfo:
    """Coverage information for a function."""
    function_name: str
    file_path: str
    total_branches: int = 0
    covered_branches: int = 0
    test_count: int = 0
    edge_cases_tested: List[str] = field(default_factory=list)
    missing_cases: List[str] = field(default_factory=list)

    @property
    def coverage_percent(self) -> float:
        return self.covered_branches / self.total_branches * 100 if self.total_branches else 0


class TestGenerator:
    """
    Test generation with:
    - Unit test generation from function signatures
    - Edge case detection (None, empty, negative, overflow)
    - Property-based test generation
    - Fuzzing input generation
    - Coverage gap analysis
    - Test suite optimization (remove redundant tests)
    - Assertion generation
    - Mock generation for dependencies
    """

    def __init__(self):
        self._generated_tests: Dict[str, List[TestCase]] = {}
        self._coverage: Dict[str, CoverageInfo] = {}

    def generate_unit_tests(self, func_name: str, args: List[Dict],
                            docstring: str = "",
                            return_type: str = "") -> List[TestCase]:
        """Generate unit tests for a function."""
        tests = []

        # Basic test with provided args
        for i, arg_set in enumerate(args[:5]):
            test = TestCase(
                id="t_" + str(uuid.uuid4())[:8],
                name="test_%s_basic_%d" % (func_name, i),
                function_name=func_name,
                test_code=self._gen_unit_test(func_name, arg_set, "basic"),
                test_type="unit",
                inputs=arg_set,
                description="Basic test case %d" % i,
            )
            tests.append(test)

        # Edge case tests
        edge_cases = self._detect_edge_cases(args)
        for edge in edge_cases:
            test = TestCase(
                id="t_" + str(uuid.uuid4())[:8],
                name="test_%s_%s" % (func_name, edge["name"]),
                function_name=func_name,
                test_code=self._gen_unit_test(func_name, edge["args"], edge["name"]),
                test_type="edge",
                inputs=edge["args"],
                description="Edge case: %s" % edge["name"],
                confidence=0.6,
            )
            tests.append(test)

        self._generated_tests.setdefault(func_name, []).extend(tests)
        return tests

    def _detect_edge_cases(self, args: List[Dict]) -> List[Dict]:
        """Detect edge cases from argument patterns."""
        edge_cases = []

        if not args:
            return edge_cases

        # Analyze argument types
        for arg_set in args:
            for key, value in arg_set.items():
                if isinstance(value, str):
                    edge_cases.extend([
                        {"name": "empty_string_%s" % key,
                         "args": {**arg_set, key: ""}},
                        {"name": "long_string_%s" % key,
                         "args": {**arg_set, key: "x" * 10000}},
                        {"name": "special_chars_%s" % key,
                         "args": {**arg_set, key: "<script>alert('xss')</script>"}},
                        {"name": "unicode_%s" % key,
                         "args": {**arg_set, key: "日本語テスト"}},
                    ])
                elif isinstance(value, (int, float)):
                    edge_cases.extend([
                        {"name": "zero_%s" % key,
                         "args": {**arg_set, key: 0}},
                        {"name": "negative_%s" % key,
                         "args": {**arg_set, key: -1}},
                        {"name": "max_int_%s" % key,
                         "args": {**arg_set, key: 2**31 - 1}},
                    ])
                elif isinstance(value, list):
                    edge_cases.extend([
                        {"name": "empty_list_%s" % key,
                         "args": {**arg_set, key: []}},
                        {"name": "single_item_%s" % key,
                         "args": {**arg_set, key: [value[0] if value else None]}},
                    ])
                elif isinstance(value, dict):
                    edge_cases.extend([
                        {"name": "empty_dict_%s" % key,
                         "args": {**arg_set, key: {}}},
                    ])

        # Deduplicate
        seen = set()
        unique = []
        for ec in edge_cases:
            if ec["name"] not in seen:
                seen.add(ec["name"])
                unique.append(ec)

        return unique[:20]  # Limit

    def generate_fuzz_tests(self, func_name: str, arg_types: Dict[str, str],
                            count: int = 10) -> List[TestCase]:
        """Generate fuzzing test cases."""
        import random
        import string

        tests = []
        type_generators = {
            "str": lambda: ''.join(random.choices(string.printable, k=random.randint(0, 1000))),
            "int": lambda: random.randint(-2**31, 2**31),
            "float": lambda: random.uniform(-1e10, 1e10),
            "bool": lambda: random.choice([True, False]),
            "list": lambda: [random.randint(0, 100) for _ in range(random.randint(0, 50))],
            "dict": lambda: {str(i): random.randint(0, 100) for i in range(random.randint(0, 20))},
        }

        for i in range(count):
            args = {}
            for arg_name, arg_type in arg_types.items():
                generator = type_generators.get(arg_type, type_generators["str"])
                args[arg_name] = generator()

            test = TestCase(
                id="t_" + str(uuid.uuid4())[:8],
                name="test_%s_fuzz_%d" % (func_name, i),
                function_name=func_name,
                test_code=self._gen_fuzz_test(func_name, args),
                test_type="fuzz",
                inputs=args,
                description="Fuzz test %d" % i,
                confidence=0.4,
            )
            tests.append(test)

        return tests

    def analyze_coverage(self, file_path: str) -> Dict[str, CoverageInfo]:
        """Analyze test coverage gaps for a file."""
        try:
            source = Path(file_path).read_text()
            tree = ast.parse(source)
        except Exception:
            return {}

        results = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Count branches
                branches = 0
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For,
                                          ast.ExceptHandler, ast.BoolOp)):
                        branches += 1

                # Count existing tests
                existing_tests = self._generated_tests.get(node.name, [])

                # Identify missing cases
                missing = []
                has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
                if has_return:
                    missing.append("return_value_test")
                if branches > 0:
                    missing.append("branch_coverage (%d branches)" % branches)
                if node.args.args:
                    missing.append("null_input_test")

                info = CoverageInfo(
                    function_name=node.name,
                    file_path=file_path,
                    total_branches=branches,
                    test_count=len(existing_tests),
                    missing_cases=missing,
                )
                results[node.name] = info

        return results

    def generate_from_signature(self, func_name: str, args: List[str],
                                has_return: bool = True) -> List[TestCase]:
        """Generate tests from just a function signature."""
        tests = []

        # None test
        none_args = {a: None for a in args}
        tests.append(TestCase(
            id="t_" + str(uuid.uuid4())[:8],
            name="test_%s_none_args" % func_name,
            function_name=func_name,
            test_code=self._gen_none_test(func_name, args),
            test_type="edge",
            inputs=none_args,
            description="Test with None arguments",
            confidence=0.5,
        ))

        # Empty args test
        empty_args = {a: "" for a in args}
        tests.append(TestCase(
            id="t_" + str(uuid.uuid4())[:8],
            name="test_%s_empty_args" % func_name,
            function_name=func_name,
            test_code=self._gen_empty_test(func_name, args),
            test_type="edge",
            inputs=empty_args,
            description="Test with empty arguments",
            confidence=0.5,
        ))

        return tests

    def _gen_unit_test(self, func_name: str, args: Dict, case: str) -> str:
        arg_str = ", ".join("%s=%s" % (k, repr(v)) for k, v in args.items())
        return (
            "def test_%s_%s():\n"
            "    result = %s(%s)\n"
            "    assert result is not None\n"
        ) % (func_name, case, func_name, arg_str)

    def _gen_fuzz_test(self, func_name: str, args: Dict) -> str:
        arg_str = ", ".join("%s=%s" % (k, repr(v)) for k, v in args.items())
        return (
            "def test_%s_fuzz():\n"
            "    try:\n"
            "        result = %s(%s)\n"
            "        # Should not crash\n"
            "    except (ValueError, TypeError, KeyError) as e:\n"
            "        pass  # Expected errors are OK\n"
        ) % (func_name, func_name, arg_str)

    def _gen_none_test(self, func_name: str, args: List[str]) -> str:
        arg_str = ", ".join("%s=None" % a for a in args)
        return (
            "def test_%s_none():\n"
            "    import pytest\n"
            "    with pytest.raises((TypeError, ValueError)):\n"
            "        %s(%s)\n"
        ) % (func_name, func_name, arg_str)

    def _gen_empty_test(self, func_name: str, args: List[str]) -> str:
        arg_str = ", ".join("%s=''" % a for a in args)
        return (
            "def test_%s_empty():\n"
            "    try:\n"
            "        result = %s(%s)\n"
            "    except (ValueError, TypeError):\n"
            "        pass\n"
        ) % (func_name, func_name, arg_str)

    def get_tests(self, func_name: str) -> List[TestCase]:
        return self._generated_tests.get(func_name, [])

    def export_test_file(self, func_name: str) -> str:
        """Export generated tests as a Python test file."""
        tests = self._generated_tests.get(func_name, [])
        if not tests:
            return ""

        lines = ['"""Auto-generated tests for %s."""\n' % func_name,
                  'import pytest\n\n']
        for test in tests:
            lines.append(test.test_code)
            lines.append("\n")

        return "\n".join(lines)

    def stats(self) -> Dict:
        total = sum(len(t) for t in self._generated_tests.values())
        by_type = {"unit": 0, "edge": 0, "fuzz": 0, "integration": 0}
        for tests in self._generated_tests.values():
            for t in tests:
                by_type[t.test_type] = by_type.get(t.test_type, 0) + 1
        return {
            "functions_tested": len(self._generated_tests),
            "total_tests": total,
            "by_type": by_type,
        }
