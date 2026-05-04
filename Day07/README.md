# Day 07 — Image Generation with OpenAI

## Overview

This demo extends the Day 06 multi-turn chat agent to also generate images.
A single terminal application calls two OpenAI APIs in sequence:

1. **Chat Completions API** (`gpt-4.1`) — generates a title, 80-word summary, and 3 highlights, plus an embedded image configuration block.
2. **Images API** (`gpt-image-1`) — uses that configuration to generate a PNG and saves it locally.

The conversation history is preserved across turns (same multi-turn pattern as Day 05/06), so you can build on previous responses or compare outputs side by side.

---

## Setup

```bash
# 1. Enter the Day07 folder
cd Day07

# 2. Copy the environment template and add your key
cp .env.example .env
# Edit .env and replace sk-your-key-here with your real key

# 3. Install dependencies (inside the project virtual environment)
pip install -r requirements.txt

# 4. Run the app
python content_agent.py
```

> **Note:** `gpt-image-1` requires a verified OpenAI organisation. If you see a permission error, check your account at platform.openai.com.

---

## How to Run

Start the app and type any topic at the prompt:

```
You: deep sea bioluminescent creatures
```

The agent will:
1. Return a **title**, **80-word summary**, and **3 bullet highlights**
2. Extract an image configuration from its own reply
3. Call `gpt-image-1` and save a PNG to `generated_outputs/`

### Built-in commands

| Command    | What it does                                                        |
|------------|---------------------------------------------------------------------|
| `variants` | Re-generates the last image at **low**, **medium**, and **high** quality |
| `history`  | Prints all conversation turns (image blocks are collapsed)          |
| `cost`     | Shows total input and output token usage for the session            |
| `clear`    | Resets conversation history (starts a fresh session)                |
| `quit`     | Exits the app                                                       |

---

## Parameter Reference

| Parameter       | Values                              | Purpose                               |
|-----------------|-------------------------------------|---------------------------------------|
| `model`         | `gpt-image-1`                       | Image generation model                |
| `prompt`        | string                              | Visual description (no text/labels)   |
| `size`          | `1024x1024`, `1536x1024`, `1024x1536` | Square, landscape, portrait         |
| `quality`       | `low`, `medium`, `high`             | Detail level — affects cost and time  |
| `output_format` | `png`, `jpeg`, `webp`               | Output file format (demo uses `png`)  |
| `n`             | 1 to 10                             | Number of images per call             |

---

## How the Image Prompt Block Works

The system prompt instructs the agent to end every reply with a structured JSON block wrapped in XML-style tags:

```
<image_prompt>
{
  "prompt": "A glowing anglerfish drifting through pitch-black deep ocean water ...",
  "size": "1536x1024",
  "quality": "medium",
  "style_notes": "Landscape orientation suits the vast empty darkness of the deep sea"
}
</image_prompt>
```

`content_agent.py` uses a regex to extract this block and passes the values directly to `client.images.generate()`. If the block is missing or malformed, the app prints a warning and skips image generation for that turn.

---

## File Structure

```
Day07/
├── content_agent.py      # Main application (single file)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── README.md             # This file
└── generated_outputs/    # Created automatically — PNGs saved here
```

---

## Demo Flow (for class)

### Step 1 — Weak prompt (observe vague output)
```
You: nature
```
Notice the image description is generic. The generated PNG will lack detail and visual interest.

### Step 2 — Structured prompt (observe improvement)
```
You: bioluminescent deep sea creatures in the midnight zone, national geographic style
```
The richer topic gives the agent more to work with. Compare the summary depth and the image prompt specificity with Step 1.

### Step 3 — Change only quality
```
You: variants
```
The `variants` command re-generates the **same prompt** at `low`, `medium`, and `high` quality. Open the three PNGs side by side and compare sharpness, detail, and generation time.

### Step 4 — Try quality manually
Type any topic, then type `variants` immediately after. This shows the quality parameter in isolation — the prompt is identical, only quality changes.

### Step 5 — Demonstrate multi-turn memory
```
You: ancient Roman architecture
You: now focus only on the Colosseum at night
You: history
```
`history` shows the full conversation. The agent remembered "ancient Roman architecture" when you said "now focus only on the Colosseum" — this is the multi-turn context window in action.

---

## Lab Tasks

1. **Prompt engineering** — Type the same topic twice: once with one word, once with a detailed description. Compare the image prompts the agent generated. What changed?

2. **Quality trade-off** — After any topic, type `variants`. Compare the three output files. Which differences do you notice? When would you use `low`?

3. **Size selection** — Modify the system prompt to always request portrait orientation (`1024x1536`). Re-run a topic and observe how the composition changes.

4. **Parameter override** — In `content_agent.py`, hard-code `quality="high"` in the `generate_image()` call. What is the trade-off?

5. **Token cost awareness** — Run three topics, then type `cost`. Where do most tokens come from? Check the per-image token counts printed after each generation.

6. **Conversation memory** — Run two related topics in sequence (e.g., "rainforest" then "zoom into the forest canopy at sunrise"). Type `history` and trace how the agent used the prior context.

---

## Common Errors

| Error                          | Cause                                      | Fix                                       |
|--------------------------------|--------------------------------------------|-------------------------------------------|
| `OPENAI_API_KEY not found`     | Missing or empty `.env` file               | Run `cp .env.example .env` and add your key |
| `AuthenticationError`          | Invalid API key                            | Verify the key at platform.openai.com     |
| `RateLimitError`               | Too many requests                          | Wait a few seconds and retry              |
| `Permission denied` (images)   | `gpt-image-1` not enabled for your org     | Check organisation settings              |
| No image saved                 | Agent did not return `<image_prompt>` block | Check the raw reply with `history`       |
| JSON parse warning             | Agent returned malformed JSON in the block | Re-run the topic; adjust the system prompt if persistent |
