#!/usr/bin/env python3
"""Debug script: inspect raw DeepSeek response to diagnose score extraction failure.

This script makes ONE judge call against the simplest rubric (compliance_judge,
3 criteria) and prints the RAW LLM response -- both tool_calls and text content
blocks -- so we can see exactly what DeepSeek returns and why the framework
fails to extract scores.
"""

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; API keys must be in environment

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harn_ai.models import get_model
from harn_ai.stream import complete_simple
from harn_ai.types import (
    Context,
    SimpleStreamOptions,
    UserMessage,
)

from examples.compliance_judge import compliance_judge
from rubrify.codecs.json_codec import (
    build_judgment_tool,
    parse_judgment_json,
    validate_judgment_output,
    ParseError,
)
from rubrify.codecs.xml_codec import render_rubric_xml

# ── Configuration ──────────────────────────────────────────────────
RESPONSE_TEXT = (
    "The bridge collapsed because the steel bolts corroded over twenty years "
    "of exposure to salt air. Three engineers inspected the site. They found "
    "cracks in fourteen support beams. The city closed the road the same day."
)


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


async def main():
    # ── Step 1: Setup ──────────────────────────────────────────────
    separator("1. MODEL AND RUBRIC SETUP")

    # First, demonstrate the actual bug: deepseek-chat no longer exists
    bad_model = get_model("deepseek", "deepseek-chat")
    print(f"get_model('deepseek', 'deepseek-chat') => {bad_model}")
    if bad_model is None:
        print("  ^^^ THIS IS None! The model ID 'deepseek-chat' no longer exists.")
        print("  This None gets passed to complete_simple() and causes silent failures.")
        from harn_ai.models import get_models
        print(f"  Available deepseek models:")
        for m in get_models("deepseek"):
            print(f"    - {m.id} (api={m.api}, provider={m.provider})")

    # Use an actually available model
    model = get_model("deepseek", "deepseek-v4-flash")
    if model is None:
        print("FATAL: get_model('deepseek', 'deepseek-v4-flash') also returned None")
        return
    print(f"Model: {model.id}")
    print(f"  api:      {model.api}")
    print(f"  provider: {model.provider}")
    print(f"  baseUrl:  {model.baseUrl}")
    print(f"  reasoning: {model.reasoning}")

    result = compliance_judge()
    bundle = result.bundle
    criterion = bundle.rubric.criteria[0]  # C1 - Directness
    print(f"\nRubric: {bundle.rubric.meta.name}")
    print(f"Target criterion: {criterion.id} - {criterion.title}")

    # ── Step 2: Build the tool ─────────────────────────────────────
    separator("2. TOOL SCHEMA (what we send to DeepSeek)")

    tool = build_judgment_tool(bundle)
    schema = tool.parameters_json_schema()
    print(f"Tool name: {tool.name}")
    print(f"Tool description: {tool.description[:80]}...")
    print(f"Tool parameters JSON schema:")
    print(json.dumps(schema, indent=2))

    # ── Step 3: Build the prompts (same as executor.py) ────────────
    separator("3. PROMPTS")

    system_prompt = render_rubric_xml(bundle)
    for constraint in bundle.output_constraints:
        system_prompt += f"\n\nCONSTRAINT [{constraint.id}]: {constraint.description}"
    print(f"System prompt length: {len(system_prompt)} chars")
    print(f"System prompt (first 500 chars):\n{system_prompt[:500]}...")

    user_prompt = (
        f"Evaluate the following response for criterion [{criterion.id}] {criterion.title}.\n\n"
        f"Criterion description: {criterion.description}\n\n"
        f"<response_under_test>\n{RESPONSE_TEXT}\n</response_under_test>\n\n"
        f"Score this response on criterion {criterion.id} using the scale and anchors defined "
        f"in the rubric above. Call the submit_judgment tool with your evaluation. "
        f"Focus exclusively on {criterion.id}. Ignore other criteria."
    )
    print(f"\nUser prompt:\n{user_prompt}")

    # ── Step 4: Make the raw LLM call ──────────────────────────────
    separator("4. RAW LLM CALL (complete_simple)")

    context = Context(
        systemPrompt=system_prompt,
        messages=[
            UserMessage(
                role="user",
                content=user_prompt,
                timestamp=int(time.time() * 1000),
            ),
        ],
        tools=[tool],
    )

    opts = SimpleStreamOptions(
        apiKey=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0.0,
        maxTokens=2048,
    )

    print("Calling complete_simple(model, context, opts)...")
    print("  (this will take a few seconds)")
    try:
        result = await complete_simple(model, context, opts)
    except Exception as e:
        print(f"\nFATAL: complete_simple raised: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # ── Step 5: Inspect the raw response ───────────────────────────
    separator("5. RAW RESPONSE INSPECTION")

    print(f"stopReason:    {result.stopReason}")
    print(f"errorMessage:  {result.errorMessage}")
    print(f"model:         {result.model}")
    print(f"responseModel: {result.responseModel}")
    print(f"provider:      {result.provider}")
    print(f"api:           {result.api}")
    print(f"\nUsage:")
    print(f"  input:       {result.usage.input}")
    print(f"  output:      {result.usage.output}")
    print(f"  totalTokens: {result.usage.totalTokens}")

    print(f"\nContent blocks: {len(result.content)}")
    for i, block in enumerate(result.content):
        print(f"\n--- Content block [{i}] ---")
        print(f"  type: {block.type}")

        if block.type == "text":
            print(f"  text length: {len(block.text)}")
            print(f"  text content:")
            print(f"  >>>{block.text}<<<")

        elif block.type == "toolCall":
            print(f"  id:   {block.id}")
            print(f"  name: {block.name}")
            print(f"  arguments type: {type(block.arguments).__name__}")
            print(f"  arguments:")
            print(f"  {json.dumps(block.arguments, indent=4)}")

        elif block.type == "thinking":
            print(f"  thinking length: {len(block.thinking)}")
            print(f"  thinking (first 300 chars): {block.thinking[:300]}")

        else:
            print(f"  (unknown block type, dumping): {block}")

    # ── Step 6: Trace the extraction logic ─────────────────────────
    separator("6. EXTRACTION LOGIC TRACE")

    # Check Strategy 1: tool call extraction
    tool_call_found = False
    for block in result.content:
        if block.type == "toolCall":
            print(f"[Strategy 1] Found toolCall block:")
            print(f"  name: {block.name}")
            print(f"  expected name: 'submit_judgment'")
            print(f"  name match: {block.name == 'submit_judgment'}")
            if block.name == "submit_judgment":
                tool_call_found = True
                print(f"  arguments: {json.dumps(block.arguments, indent=4)}")
                validated, warnings = validate_judgment_output(block.arguments, bundle)
                print(f"  validated: {validated}")
                print(f"  warnings: {warnings}")
                if validated:
                    cs = getattr(validated, "criterion_scores", None)
                    print(f"  criterion_scores: {cs}")
                    if cs:
                        c1_val = getattr(cs, "C1", "MISSING")
                        c2_val = getattr(cs, "C2", "MISSING")
                        c3_val = getattr(cs, "C3", "MISSING")
                        print(f"    C1={c1_val}, C2={c2_val}, C3={c3_val}")
            else:
                print(f"  MISMATCH: tool name '{block.name}' != 'submit_judgment'")

    if not tool_call_found:
        print("[Strategy 1] NO toolCall block with name 'submit_judgment' found!")

    # Check Strategy 2: text fallback
    raw_text = ""
    for block in result.content:
        if block.type == "text":
            raw_text += block.text

    if raw_text.strip():
        print(f"\n[Strategy 2] Text fallback - raw text ({len(raw_text)} chars):")
        print(f"  >>>{raw_text[:1000]}<<<")
        try:
            parsed = parse_judgment_json(raw_text)
            print(f"  Parsed successfully: {json.dumps(parsed, indent=4)}")
            validated, warnings = validate_judgment_output(parsed, bundle)
            print(f"  validated: {validated}")
            print(f"  warnings: {warnings}")
        except ParseError as e:
            print(f"  ParseError: {e}")
            print(f"  raw_text for parse: >>>{e.raw_text[:500]}<<<")
    else:
        print(f"\n[Strategy 2] No text content to fall back to")

    # ── Step 7: Final diagnosis ────────────────────────────────────
    separator("7. DIAGNOSIS")

    if not result.content:
        print("DIAGNOSIS: Response has NO content blocks at all.")
        print("  This means the LLM returned an empty response.")
    elif tool_call_found:
        print("DIAGNOSIS: Tool call WAS found and matched 'submit_judgment'.")
        print("  If scores are still 0, the issue is in validation/extraction.")
    elif any(b.type == "toolCall" for b in result.content):
        misnamed = [b for b in result.content if b.type == "toolCall"]
        print(f"DIAGNOSIS: Tool call blocks exist but with WRONG NAME(S):")
        for b in misnamed:
            print(f"  - name='{b.name}' (expected 'submit_judgment')")
        print("  The tool call is being generated but the name doesn't match.")
    elif raw_text.strip():
        print("DIAGNOSIS: DeepSeek returned TEXT instead of a tool call.")
        print("  The text fallback path will be used.")
        print("  Check if the text contains valid JSON with criterion_scores.")
    else:
        print("DIAGNOSIS: Response has content blocks but no tool calls and no text.")
        print("  Block types present:", [b.type for b in result.content])


if __name__ == "__main__":
    asyncio.run(main())
