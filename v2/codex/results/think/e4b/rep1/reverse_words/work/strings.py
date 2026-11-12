def reverse_words(text):
    """Reverse words in the text with and without punctuation."""
    # Split text into words and strip punctuation
    words = [w.strip() for w in text.split()]
    
    # If words have no punctuation, just return them
    if words and all(w for w in words if w):
        return ' '.join(reversed(words))
    
    # Sort words to check if they're separated
    words_sorted = sorted(words)
    
    if words_sorted == words_sorted[::-1]:
        # All words are reversed, join them
        return ' '.join(reversed(words))
    
    # Check if words are separated by punctuation
    # Get all non-space characters
    all_chars = words[0] + ' ' + words[1]
    raw_chars = all_chars.lower().isalnum()
    
    if raw_chars == True:
        # Words have punctuation, reverse them in order
        return ' '.join(reversed(words))
    
    # Words are separated, return joined list
    return reversed(words)
