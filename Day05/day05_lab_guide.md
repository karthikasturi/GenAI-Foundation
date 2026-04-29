## Day 05 Lab: OpenAI API Integration

### Lab Exercises

**1. API Key Handling**
- Load your OpenAI API key from `.env` using `python-dotenv`.
- Validate that the key is loaded (do not print the full key).

**2. Send a Prompt and Print the Response**
- Use the OpenAI SDK to send a simple prompt (e.g., summarize a sentence).
- Print the response text.

**3. Tune Parameters**
- Change `temperature` and `max_tokens` values and observe the effect on output.

**4. Structured Output**
- Request a JSON output with summary and sentiment fields.
- Parse and print the JSON result.

**5. Error Handling**
- Handle invalid key and rate limit errors gracefully.

**6. Mini Use Case: Email Rewriter**
- Task: Read an email from `email_sample.txt`.
- Write a prompt to rewrite the email (e.g., make it more polite, concise, or formal).
- Send the prompt and email to the API and print the rewritten version.
- Try different rewrite styles (polite, concise, formal, friendly) and compare outputs.
- No sample code provided—implement your own solution using the earlier exercises as reference.

---

def load_api_key():
def send_prompt(prompt, model="text-davinci-003", max_tokens=50, temperature=0.5):
def summarize_ticket(ticket_text):

## Step-by-Step Participant Exercises

**Exercise 1: API Key Handling**
- Load your OpenAI API key from `.env` using `python-dotenv`.
- Validate that the key is loaded (do not print the full key).

**Exercise 2: Send a Prompt and Print the Response**
- Use the OpenAI SDK to send a simple prompt (e.g., summarize a sentence).
- Print the response text.

**Exercise 3: Tune Parameters**
- Change `temperature` and `max_tokens` values and observe the effect on output.

**Exercise 4: Structured Output**
- Request a JSON output with summary and sentiment fields.
- Parse and print the JSON result.

**Exercise 5: Error Handling**
- Handle invalid key and rate limit errors gracefully.

**Exercise 6: Mini Use Case: Email Rewriter**
- Read an email from `email_sample.txt`.
- Write a prompt to rewrite the email (e.g., make it more polite, concise, or formal).
- Send the prompt and email to the API and print the rewritten version.
- Try different rewrite styles (polite, concise, formal, friendly) and compare outputs.
- No sample code provided—implement your own solution using the earlier exercises as reference.

---

## 15. Common Errors and Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| Invalid API key | Wrong or missing key in .env | Check .env and reload |
| Rate limit exceeded | Too many requests | Wait and retry, add delay |
| Timeout | Network issues | Check connection, retry |
| JSON decode error | Output not valid JSON | Add clearer format instructions |
| ModuleNotFoundError | Missing package | Run `pip install -r requirements.txt` |

---
