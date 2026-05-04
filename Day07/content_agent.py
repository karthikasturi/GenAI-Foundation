#!/usr/bin/env python3
"""
Day 07 Demo: Image Generation with OpenAI
==========================================
Extends the Day 06 multi-turn chat agent to also generate images using
the OpenAI Images API (gpt-image-1).

The agent returns structured text content (title, summary, highlights)
plus an embedded <image_prompt> JSON block that drives the image call.

Setup:
  cp .env.example .env          # add your OPENAI_API_KEY
  pip install -r requirements.txt
  python content_agent.py
"""

import os
import re
import json
import base64
import datetime
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
TEXT_MODEL   = "gpt-4.1"        # Chat completions model for text generation
IMAGE_MODEL  = "gpt-image-1"    # Image generation model
OUTPUT_DIR   = "generated_outputs"  # Folder where PNGs will be saved

# ---------------------------------------------------------------------------
# System prompt
# The agent MUST end every reply with a well-formed <image_prompt> block.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a creative content assistant for a visual media team.
When the user provides a topic, respond with ALL of the following sections in this exact order:

1. **Title** — a concise, engaging title for the topic.
2. **Summary** — exactly 80 words describing the topic clearly and vividly.
3. **Highlights** — exactly 3 bullet points, each starting with "•".
4. An image prompt block wrapped in XML-style tags, exactly like this:

<image_prompt>
{
  "prompt": "detailed visual description of the scene",
  "size": "1536x1024",
  "quality": "medium",
  "style_notes": "reason for this visual direction"
}
</image_prompt>

Rules for the image prompt:
- The "prompt" field must include: subject, setting, artistic style, lighting, and composition.
- Never include text, words, typography, labels, signs, or captions in the "prompt" field.
- Choose "size" based on the topic: landscape (1536x1024), portrait (1024x1536), or square (1024x1024).
- Valid "quality" values: low | medium | high. Default to medium.
- Always end your reply with the closing </image_prompt> tag — no text after it."""

# ---------------------------------------------------------------------------
# Token tracking — running totals across all turns in this session
# ---------------------------------------------------------------------------
total_input_tokens  = 0
total_output_tokens = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_client() -> OpenAI:
    """Load OPENAI_API_KEY from .env and return an authenticated OpenAI client."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Create a .env file with OPENAI_API_KEY=sk-... or export the variable."
        )
    return OpenAI(api_key=api_key)


def ensure_output_dir() -> None:
    """Create the generated_outputs folder if it does not already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def sanitise_filename(text: str) -> str:
    """Convert arbitrary user input into a safe filename fragment (max 50 chars)."""
    # Keep only alphanumeric characters and spaces/hyphens
    safe = re.sub(r"[^a-zA-Z0-9 \-]", "", text).strip()
    # Replace runs of whitespace with a single underscore
    safe = re.sub(r"\s+", "_", safe)
    return safe[:50]


def extract_image_config(text: str) -> dict | None:
    """
    Find and parse the <image_prompt>…</image_prompt> block inside the agent reply.

    Returns a dict with keys: prompt, size, quality, style_notes.
    Returns None if the block is absent or the JSON is malformed.
    """
    # Use DOTALL so '.' matches newlines inside the block
    pattern = r"<image_prompt>\s*(\{.*?\})\s*</image_prompt>"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        print("\n[Warning] No <image_prompt> block found in the agent response.")
        return None

    try:
        config = json.loads(match.group(1))
        return config
    except json.JSONDecodeError as exc:
        print(f"\n[Warning] Could not parse image prompt JSON: {exc}")
        return None


def print_text_content(reply: str) -> None:
    """
    Print the readable portion of the agent reply.
    Strips the <image_prompt> block so the terminal stays clean.
    """
    clean = re.sub(
        r"<image_prompt>.*?</image_prompt>", "", reply, flags=re.DOTALL
    ).strip()
    print("\n" + clean + "\n")


# ---------------------------------------------------------------------------
# Core: text generation (same multi-turn pattern as Day 05 / Day 06)
# ---------------------------------------------------------------------------

def chat(client: OpenAI, history: list, user_message: str) -> str:
    """
    Append user_message to history, call the Chat Completions API,
    store the assistant reply, and return it.

    The system prompt is prepended on every call — it is NOT stored in history,
    which keeps the history list portable and easy to inspect.
    """
    global total_input_tokens, total_output_tokens

    # Add the new user turn to conversation history
    history.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            # System prompt is sent fresh every turn so it stays authoritative
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            max_tokens=1024,
            temperature=0.7,   # Some creativity while staying on-topic
        )
    except AuthenticationError:
        print("[Error] Invalid API key. Check your .env file.")
        history.pop()          # Remove the user turn we just added
        return ""
    except RateLimitError:
        print("[Error] Rate limit reached. Wait a moment and try again.")
        history.pop()
        return ""
    except APIConnectionError:
        print("[Error] Could not reach the OpenAI API. Check your internet connection.")
        history.pop()
        return ""

    reply = response.choices[0].message.content.strip()

    # Persist the assistant reply so future turns have full context
    history.append({"role": "assistant", "content": reply})

    # Accumulate token usage for the cost command
    if response.usage:
        total_input_tokens  += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens

    return reply


# ---------------------------------------------------------------------------
# Core: image generation
# ---------------------------------------------------------------------------

def generate_image(
    client: OpenAI,
    config: dict,
    filename_hint: str,
) -> str | None:
    """
    Call the Images API (gpt-image-1) using parameters from the agent config block.

    Parameters (all sourced from config):
      prompt        — visual description string
      size          — "1024x1024" | "1536x1024" | "1024x1536"
      quality       — "low" | "medium" | "high"
      output_format — hardcoded to "png" for this demo
      n             — hardcoded to 1 (one image per call)

    Returns the saved file path, or None on failure.
    """
    prompt  = config.get("prompt", "")
    size    = config.get("size", "1536x1024")
    quality = config.get("quality", "medium")

    # Validate size — fall back to default if the model returned something unexpected
    valid_sizes = {"1024x1024", "1536x1024", "1024x1536"}
    if size not in valid_sizes:
        print(f"[Warning] Unknown size '{size}' — using default 1536x1024.")
        size = "1536x1024"

    print(f"\n[Image] Generating image …")
    print(f"  model  : {IMAGE_MODEL}")
    print(f"  size   : {size}")
    print(f"  quality: {quality}")
    print(f"  prompt : {prompt[:100]}{'…' if len(prompt) > 100 else ''}")

    try:
        response = client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size=size,
            quality=quality,
            output_format="png",   # We always save as PNG in this demo
            n=1,                   # One image per call
        )
    except Exception as exc:
        print(f"[Error] Image generation failed: {exc}")
        return None

    # The Images API returns base64-encoded data in data[0].b64_json
    image_bytes = base64.b64decode(response.data[0].b64_json)

    # Build a timestamped filename so files never overwrite each other
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{sanitise_filename(filename_hint)}_{timestamp}.png"
    filepath  = os.path.join(OUTPUT_DIR, filename)

    ensure_output_dir()
    with open(filepath, "wb") as fh:
        fh.write(image_bytes)

    print(f"[Image] Saved → {filepath}")

    # gpt-image-1 exposes token usage on the response object
    if hasattr(response, "usage") and response.usage:
        u = response.usage
        img_input  = getattr(u, "input_tokens",  "?")
        img_output = getattr(u, "output_tokens", "?")
        print(f"[Usage] Image tokens — input: {img_input}  output: {img_output}")

    return filepath


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------

def cmd_variants(
    client: OpenAI,
    last_config: dict | None,
    filename_hint: str,
) -> None:
    """
    Generate the same image prompt at low, medium, and high quality side-by-side.
    Useful for comparing quality trade-offs in class.
    """
    if not last_config:
        print("[Info] No image prompt available yet. Run a topic first.")
        return

    print("\n[Variants] Generating low / medium / high quality versions …")
    for quality_level in ("low", "medium", "high"):
        cfg = {**last_config, "quality": quality_level}
        generate_image(client, cfg, f"{filename_hint}_{quality_level}")


def cmd_history(history: list) -> None:
    """Print all conversation turns, collapsing the image_prompt blocks."""
    if not history:
        print("[Info] Conversation history is empty.")
        return

    print("\n" + "=" * 60)
    print("  Conversation History")
    print("=" * 60)

    for idx, turn in enumerate(history, start=1):
        role    = turn["role"].upper()
        content = turn["content"]

        # Collapse the image_prompt block so the history is readable
        content_display = re.sub(
            r"<image_prompt>.*?</image_prompt>",
            "<image_prompt> … </image_prompt>",
            content,
            flags=re.DOTALL,
        )
        print(f"\n[{idx}] {role}:\n{content_display}")

    print("\n" + "=" * 60)


def cmd_cost() -> None:
    """Display cumulative token usage for this session."""
    total = total_input_tokens + total_output_tokens
    print("\n[Cost] Token usage this session:")
    print(f"  Input  tokens : {total_input_tokens}")
    print(f"  Output tokens : {total_output_tokens}")
    print(f"  Total         : {total}")
    print(
        "  Note: gpt-image-1 image tokens are tracked separately above each image."
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Day 07 — Content + Image Generation Agent")
    print(f"  Text model : {TEXT_MODEL}")
    print(f"  Image model: {IMAGE_MODEL}")
    print("=" * 60)
    print("Commands: variants | history | cost | clear | quit")
    print("Enter any topic to generate content and an image.\n")

    # Initialise the OpenAI client once at startup
    try:
        client = load_client()
    except ValueError as exc:
        print(f"[Error] {exc}")
        return

    ensure_output_dir()

    # conversation_history holds all user/assistant turns (same pattern as Day 06)
    # The system prompt is NOT stored here — it is prepended on every API call.
    conversation_history: list[dict] = []

    last_config: dict | None = None    # Most recent image prompt config (for variants)
    last_topic:  str         = "image" # Used as the filename hint

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exit] Goodbye!")
            break

        if not user_input:
            continue

        # ----------------------------------------------------------------
        # Built-in command dispatch
        # ----------------------------------------------------------------
        lower = user_input.lower()

        if lower == "quit":
            print("[Exit] Goodbye!")
            break

        elif lower == "clear":
            conversation_history.clear()
            last_config = None
            print("[Info] Conversation history cleared.")

        elif lower == "history":
            cmd_history(conversation_history)

        elif lower == "cost":
            cmd_cost()

        elif lower == "variants":
            cmd_variants(client, last_config, last_topic)

        else:
            # ----------------------------------------------------------------
            # Normal topic turn
            # ----------------------------------------------------------------
            last_topic = user_input
            print("\n[Agent] Thinking …")

            # 1. Call gpt-4.1 to get structured text + image config
            reply = chat(client, conversation_history, user_input)
            if not reply:
                continue  # Error already printed inside chat()

            # 2. Print the human-readable text section
            print_text_content(reply)

            # 3. Extract the <image_prompt> config from the reply
            config = extract_image_config(reply)

            if config:
                # Store the config for potential re-use by the variants command
                last_config = config
                # 4. Generate and save the image
                generate_image(client, config, last_topic)
            else:
                print("[Info] Skipping image generation — no valid config found.")


if __name__ == "__main__":
    main()
