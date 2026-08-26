"""Structured code generation — builds LLM prompts and parses generated code."""

import re
from typing import Optional, Tuple
from dataclasses import dataclass

from llm.ollama_client import OllamaClient, OllamaResponse
from llm.prompts import get_prompt
from profiling.schema_detector import DatasetSchema


@dataclass
class GeneratedCode:
    """Result of LLM code generation."""
    success: bool
    code: str = ""
    raw_response: str = ""
    error: Optional[str] = None


class CodeGenerator:
    """Generates pandas/SQL code from natural language using Ollama."""

    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def generate_pandas_code(
        self,
        question: str,
        schema: DatasetSchema,
        sample_data: str,
    ) -> GeneratedCode:
        """Generate pandas code from a natural language question.

        Args:
            question: User's natural language question.
            schema: Dataset schema information.
            sample_data: String representation of first 5 rows.

        Returns:
            GeneratedCode with extracted Python code.
        """
        # Build schema string
        schema_str = self._schema_to_string(schema)

        prompt = get_prompt(
            "nl_to_code",
            schema=schema_str,
            sample=sample_data,
            question=question,
        )

        response: OllamaResponse = self.client.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=2048,
        )

        if not response.success:
            return GeneratedCode(
                success=False,
                raw_response=response.text,
                error=response.error,
            )

        code = self._extract_code_block(response.text)

        if not code:
            return GeneratedCode(
                success=False,
                raw_response=response.text,
                error="No code block found in LLM response",
            )

        # Sanitize the code
        code = self._sanitize_code(code)

        return GeneratedCode(
            success=True,
            code=code,
            raw_response=response.text,
        )

    def generate_insights(
        self,
        stats_summary: str,
    ) -> GeneratedCode:
        """Generate auto-insights from dataset statistics.

        Args:
            stats_summary: String representation of dataset statistics.

        Returns:
            GeneratedCode with a JSON array of insight strings.
        """
        prompt = get_prompt("insights", stats=stats_summary)

        response = self.client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1024,
        )

        if not response.success:
            return GeneratedCode(
                success=False,
                error=response.error,
            )

        return GeneratedCode(
            success=True,
            code=response.text,
            raw_response=response.text,
        )

    def generate_anomaly_explanation(
        self,
        anomaly_data: str,
    ) -> GeneratedCode:
        """Generate explanations for detected anomalies.

        Args:
            anomaly_data: String representation of anomaly details.

        Returns:
            GeneratedCode with JSON array of explanations.
        """
        prompt = get_prompt("anomaly_explain", anomaly_data=anomaly_data)

        response = self.client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1024,
        )

        if not response.success:
            return GeneratedCode(
                success=False,
                error=response.error,
            )

        return GeneratedCode(
            success=True,
            code=response.text,
            raw_response=response.text,
        )

    def generate_forecast_explanation(
        self,
        forecast_data: str,
    ) -> GeneratedCode:
        """Generate a narrative explanation for a time series forecast.

        Args:
            forecast_data: String representation of forecast results.

        Returns:
            GeneratedCode with JSON object explanation.
        """
        prompt = get_prompt("forecast_explain", forecast_data=forecast_data)

        response = self.client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1024,
        )

        if not response.success:
            return GeneratedCode(
                success=False,
                error=response.error,
            )

        return GeneratedCode(
            success=True,
            code=response.text,
            raw_response=response.text,
        )

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Extract Python code from a markdown code block.

        Args:
            text: LLM response text.

        Returns:
            Extracted code string, or empty string if not found.
        """
        # Try ```python ... ```
        pattern_python = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern_python, text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Try ``` ... ```
        pattern_generic = r"```\s*\n(.*?)```"
        matches = re.findall(pattern_generic, text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Try bare code (if the LLM just output code without fences)
        # Check if the text looks like Python
        lines = text.strip().split("\n")
        python_indicators = ["=", "def ", "if ", "for ", "print(", "df.", "pd."]
        if any(any(ind in line for ind in python_indicators) for line in lines[:5]):
            return text.strip()

        return ""

    @staticmethod
    def _sanitize_code(code: str) -> str:
        """Remove dangerous constructs that might have slipped through.

        Args:
            code: Raw extracted code.

        Returns:
            Sanitized code string.
        """
        # Remove any import lines that somehow got through
        lines = code.split("\n")
        safe_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            if "exec(" in stripped or "eval(" in stripped:
                continue
            if "__" in stripped and ("import" in stripped or "builtins" in stripped):
                continue
            safe_lines.append(line)

        return "\n".join(safe_lines)

    @staticmethod
    def _schema_to_string(schema: DatasetSchema) -> str:
        """Convert a DatasetSchema to a human-readable string for prompts.

        Args:
            schema: The dataset schema.

        Returns:
            Formatted schema string.
        """
        lines = [f"Rows: {schema.total_rows}, Columns: {schema.total_columns}"]
        lines.append("")

        for col in schema.columns:
            stats_str = ""
            if col.inferred_type == "numeric" and col.stats:
                stats_str = (
                    f" (mean={col.stats.get('mean', '?')}, "
                    f"std={col.stats.get('std', '?')}, "
                    f"min={col.stats.get('min', '?')}, "
                    f"max={col.stats.get('max', '?')})"
                )
            elif col.inferred_type == "categorical" and col.stats:
                top = col.stats.get("mode", "?")
                stats_str = f" (mode={top})"

            lines.append(
                f"- {col.name}: {col.inferred_type} | "
                f"non-null={col.non_null_count}/{col.non_null_count + col.null_count} | "
                f"unique={col.unique_count}"
                f"{stats_str}"
            )

        return "\n".join(lines)
