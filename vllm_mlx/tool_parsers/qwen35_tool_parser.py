# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 tool call parser for vllm-mlx.

Handles Qwen3.5's native tool calling format from chat_template.jinja:

    <tool_call>
    <function=example_function_name>
    <parameter=example_parameter_1>
    value_1
    </parameter>
    <parameter=example_parameter_2>
    multi-line value
    </parameter>
    </function>
    </tool_call>

This is distinct from the older Qwen JSON format:
    <tool_call>{"name": "func", "arguments": {...}}</tool_call>
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["qwen35", "qwen3.5"])
class Qwen35ToolParser(ToolParser):
    """
    Tool call parser for Qwen3.5 models.

    Parses the <function=name><parameter=arg>value</parameter></function>
    format that Qwen3.5 chat templates instruct models to produce.

    Handles:
    - <think>...</think> stripping (so --reasoning-parser is not needed)
    - Single and multi-parameter tool calls
    - Multi-line parameter values
    - Multiple sequential tool calls in one response
    - Both streaming and non-streaming paths

    Used with: --enable-auto-tool-choice --tool-call-parser qwen35
    """

    # Qwen3.5's chat template natively handles role="tool" messages
    # (renders them as <tool_response>...</tool_response>) and assistant
    # messages with tool_calls. Keeping native format lets the model see
    # its own tool call history in the format it was trained on, which
    # prevents it from "forgetting" it used tools in previous turns.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # Matches the outer <tool_call>...</tool_call> wrapper
    TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    # Matches <function=name>...</function> inside a tool call block
    FUNCTION_BLOCK = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)

    # Matches individual <parameter=name>value</parameter> pairs
    PARAMETER_BLOCK = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Qwen3.5 model response."""
        # Strip <think>...</think> blocks so we don't need --reasoning-parser
        cleaned_text = self.strip_think_tags(model_output)

        tool_calls = []
        tool_call_blocks = self.TOOL_CALL_BLOCK.findall(cleaned_text)

        for block in tool_call_blocks:
            func_match = self.FUNCTION_BLOCK.search(block)
            if not func_match:
                continue

            func_name = func_match.group(1).strip()
            func_body = func_match.group(2)

            # Build arguments dict from <parameter=key>value</parameter> pairs
            arguments: dict[str, Any] = {}
            for param_match in self.PARAMETER_BLOCK.finditer(func_body):
                param_name = param_match.group(1).strip()
                param_value = param_match.group(2).strip()

                # Try to parse as JSON (for structured values); fall back to string
                try:
                    arguments[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    arguments[param_name] = param_value

            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": func_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # Remove all <tool_call>...</tool_call> blocks from the content
        remaining = self.TOOL_CALL_BLOCK.sub("", cleaned_text).strip()
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=remaining if remaining else None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extract tool calls from streaming Qwen3.5 output."""
        # Pass through normal content until we see a tool call marker
        if "<tool_call>" not in current_text:
            return {"content": delta_text}

        # Accumulate until the closing tag arrives
        if "</tool_call>" not in delta_text:
            return None  # Still accumulating — suppress content output

        # Full tool call received; parse the complete accumulated text
        result = self.extract_tool_calls(current_text)
        if result.tools_called:
            return {
                "tool_calls": [
                    {
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for i, tc in enumerate(result.tool_calls)
                ]
            }

        return None
