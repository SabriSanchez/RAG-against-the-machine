# Security utilities for sanitizing credentials and performing
# static code analysis (SAST).

import ast
import re
from typing import List, Tuple


class SecuritySanitizer:
    """Scans and masks secrets and credentials in text using
    pre-compiled regex patterns."""

    PATTERNS_COMPILED = {
        "API_KEY": re.compile(
            r"(api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]"
            r"\s*['\"]([A-Za-z0-9_\-=]{8,})['\"]",
            re.IGNORECASE,
        ),
        "JWT_TOKEN": re.compile(
            r"eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*"
        ),
        "AWS_KEY": re.compile(r"(AKIA|ASIA)[0-9A-Z]{16}"),
        "PRIVATE_KEY": re.compile(r"-----BEGIN (RSA|EC|PGP|PRIVATE) KEY-----"),
    }

    @classmethod
    def sanitize(cls, text: str) -> Tuple[str, int]:
        """Scan and replace sensitive credentials in text with
        placeholder mask tokens.

        Args:
            text (str): The input string to sanitize.

        Returns:
            Tuple[str, int]: A tuple containing the sanitized text and
            the total count of detected and masked secrets.
        """
        sanitized_text = text
        total_masked = 0

        for secret_type, pattern in cls.PATTERNS_COMPILED.items():
            matches = list(pattern.finditer(sanitized_text))
            if matches:
                total_masked += len(matches)
                sanitized_text = pattern.sub(
                    f"[MASKED_{secret_type}]", sanitized_text
                )

        return sanitized_text, total_masked


class SASTScanner:
    """Performs lightweight static security audits using AST
    parsing with regex fallback."""

    REGEX_RULES_COMPILED = [
        (
            re.compile(r"\beval\s*\("),
            "CWE-95: Dangerous use of eval() detected.",
        ),
        (
            re.compile(r"\bexec\s*\("),
            "CWE-95: Dangerous use of exec() detected.",
        ),
        (
            re.compile(r"\bpickle\.loads?\b"),
            "CWE-502: Insecure dematerialization with `pickle`.",
        ),
        (
            re.compile(r"\bos\.system\s*\("),
            "CWE-78: Command injection using os.system().",
        ),
        (
            re.compile(r"shell\s*=\s*True"),
            "CWE-78: Possible command injection via `shell=True`.",
        ),
        (
            re.compile(r"\byaml\.load\s*\("),
            "CWE-20: Loading YAML unsafely without SafeLoader.",
        ),
    ]

    DANGEROUS_FUNCTIONS = {
        "eval": "CWE-95: Dangerous use of eval() detected.",
        "exec": "CWE-95: Dangerous use of exec() detected.",
        "compile": "CWE-95: Dangerous use of compile() detected.",
        "__import__": "CWE-95: Dangerous use of __import__() detected.",
    }

    DANGEROUS_ATTRIBUTES = {
        ("os", "system"): "CWE-78: Command injection using os.system().",
        ("os", "popen"): "CWE-78: Command injection using os.popen().",
        (
            "pickle",
            "loads",
        ): "CWE-502: Insecure dematerialization with `pickle`.",
        (
            "pickle",
            "load",
        ): "CWE-502: Insecure dematerialization with `pickle`.",
        ("yaml", "load"): "CWE-20: Loading YAML unsafely without SafeLoader.",
        (
            "subprocess",
            "Popen",
        ): "AST-Alert: Use of subprocess.Popen() detected.",
        (
            "subprocess",
            "call",
        ): "AST-Alert: Use of subprocess.call() detected.",
        ("subprocess", "run"): "AST-Alert: Use of subprocess.run() detected.",
    }

    @classmethod
    def scan(cls, text: str, is_python: bool = False) -> List[str]:
        """Scan text or Python source code for security vulnerabilities.

        Args:
            text (str): The source code or plain text content to scan.
            is_python (bool, optional): Whether to inspect the input as
            Python code using AST analysis. Defaults to False.

        Returns:
            List[str]: A list of warning messages indicating
            detected security issues.
        """
        warnings: List[str] = []

        if not is_python:
            cls._scan_regex(text, warnings)
        else:
            try:
                parsed_ast = ast.parse(text)
                for node in ast.walk(parsed_ast):
                    if isinstance(node, ast.Call):
                        if (
                            isinstance(node.func, ast.Name)
                            and node.func.id in cls.DANGEROUS_FUNCTIONS
                        ):
                            msg = cls.DANGEROUS_FUNCTIONS[node.func.id]
                            if msg not in warnings:
                                warnings.append(msg)

                        elif isinstance(
                            node.func, ast.Attribute
                        ) and isinstance(node.func.value, ast.Name):
                            module_name = node.func.value.id
                            func_name = node.func.attr
                            key = (module_name, func_name)
                            if key in cls.DANGEROUS_ATTRIBUTES:
                                msg = cls.DANGEROUS_ATTRIBUTES[key]
                                if msg not in warnings:
                                    warnings.append(msg)

                        for kw in node.keywords:
                            if (
                                kw.arg == "shell"
                                and isinstance(kw.value, ast.Constant)
                                and kw.value.value is True
                            ):
                                msg = (
                                    "CWE-78: Possible command injection"
                                    "via `shell=True`."
                                )
                                if msg not in warnings:
                                    warnings.append(msg)
            except SyntaxError:
                cls._scan_regex(text, warnings)

        return warnings

    @classmethod
    def _scan_regex(cls, text: str, warnings: List[str]) -> None:
        """Scan input text against regex rules and append
        missing warning messages.

        Args:
            text (str): The content to scan using compiled regex patterns.
            warnings (List[str]): The list of warnings where
            new findings are appended.
        """
        for pattern, warning_msg in cls.REGEX_RULES_COMPILED:
            if pattern.search(text) and warning_msg not in warnings:
                warnings.append(warning_msg)
