"""
Day 02 Demo: Foundations of NLP and Transformers (with pandas)
Beginner-friendly code for text preprocessing, tokenization, context, and attention intuition.
Now includes pandas DataFrame for visualizing tokens and counts.
"""

import string
import pandas as pd

# Sample multiline text block
SAMPLE_TEXT = """
  The Bank of the river was steep.
  I went to the bank to deposit money.
  Let's sit on the river bank and watch the sunset!
"""

def clean_text(text):
    """
    Lowercase, remove punctuation, and trim extra whitespace.
    """
    print("Raw text:")
    print(text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = ' '.join(text.split())
    print("\nCleaned text:")
    print(text)
    return text

def tokenize_text(text):
    """
    Split text into tokens (words).
    """
    tokens = text.split()
    print("\nTokens:")
    print(tokens)
    # Show tokens in a pandas DataFrame
    df = pd.DataFrame({'Token': tokens})
    print("\nTokens in DataFrame:")
    print(df)
    return tokens, df

def count_tokens(tokens):
    """
    Count the number of tokens and show with pandas.
    """
    count = len(tokens)
    print(f"\nNumber of tokens: {count}")
    # Show count in a DataFrame
    df = pd.DataFrame({'Count': [count]})
    print("\nToken Count in DataFrame:")
    print(df)
    return count

def compare_context_examples():
    """
    Show how 'bank' changes meaning based on context.
    """
    examples = [
        "The Bank of the river was steep.",
        "I went to the bank to deposit money.",
        "Let's sit on the river bank and watch the sunset!"
    ]
    print("\nContext Meaning Demo:")
    df = pd.DataFrame({'Example': [1, 2, 3], 'Sentence': examples})
    print(df)
    print("\nNotice how 'bank' refers to a river's edge in some sentences, and a financial institution in another.")

def attention_intuition_demo():
    """
    Highlight important words in a sentence (attention intuition).
    """
    sentence = "The quick brown fox jumps over the lazy dog."
    important_words = ["fox", "jumps", "dog"]
    print("\nAttention Intuition Demo:")
    print("Original sentence:")
    print(sentence)
    print("Highlighting important words (like attention):")
    highlighted = []
    for word in sentence.split():
        if word.strip(string.punctuation).lower() in important_words:
            highlighted.append(f"[{word.upper()}]")
        else:
            highlighted.append(word)
    print(' '.join(highlighted))
    # Show attention in a DataFrame
    df = pd.DataFrame({
        'Word': sentence.split(),
        'Is_Attended': [w.strip(string.punctuation).lower() in important_words for w in sentence.split()]
    })
    print("\nAttention DataFrame:")
    print(df)
    print("\nThis shows how attention helps focus on key words.")

def main():
    print("=== Day 02 NLP Lab Demo (with pandas) ===\n")
    # 1. Clean and tokenize sample text
    cleaned = clean_text(SAMPLE_TEXT)
    tokens, tokens_df = tokenize_text(cleaned)
    count_tokens(tokens)

    # 2. Context meaning example
    compare_context_examples()

    # 3. Attention intuition demo
    attention_intuition_demo()

    # 4. Hands-on: Try your own sentence
    print("\n--- Hands-On: Try Your Own ---")
    user_text = "The bat flew at night. He swung the bat at the ball."
    print(f"\nYour sample text:\n{user_text}")
    cleaned_user = clean_text(user_text)
    tokens_user, tokens_user_df = tokenize_text(cleaned_user)
    count_tokens(tokens_user)
    print("\nTry changing 'user_text' to your own sentence and rerun this section!")

if __name__ == "__main__":
    main()

"""
# Expected Output (Trainer Notes):

- Raw and cleaned text are printed for comparison.
- Tokens are shown as a list and in a DataFrame.
- Token count is displayed and shown in a DataFrame.
- Context examples show 'bank' with different meanings in a DataFrame.
- Attention demo highlights important words and shows a DataFrame.
- Hands-on section encourages learners to modify and observe changes.

# Troubleshooting:
- If tokens look odd, check if punctuation was removed.
- If you get errors, check for typos or missing parentheses.
"""
