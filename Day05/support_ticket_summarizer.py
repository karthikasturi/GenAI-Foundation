from utils import send_prompt
import json

def summarize_ticket(ticket_text):
    prompt = (
        "Summarize the following support ticket and label its sentiment (Positive, Negative, Neutral). "
        "Return the result as JSON with keys 'summary' and 'sentiment'.\n\n"
        f"Ticket: {ticket_text}"
    )
    raw = send_prompt(prompt, max_tokens=300, temperature=0.3, response_format={"type": "json_object"})
    return json.loads(raw)

if __name__ == "__main__":
    with open("support_ticket_samples.txt") as f:
        ticket = f.read().strip()
    result = summarize_ticket(ticket)
    print("Summary:", result["summary"])
    print("Sentiment:", result["sentiment"])
