import json
import re
import argparse
from transformers import AutoTokenizer

def extract_and_save_top_words(
    model_path: str, 
    top_n: int,
    save_path: str
):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
    
    extended_stopwords = {
        "the", "and", "of", "to", "in", "is", "that", "for", "it", "as", "was", "with", 
        "be", "by", "on", "not", "he", "i", "this", "are", "or", "his", "from", "at", 
        "which", "but", "have", "an", "had", "they", "you", "were", "their", "one", 
        "all", "we", "can", "her", "has", "there", "been", "if", "more", "when", "will", 
        "would", "who", "so", "no", "what", "up", "out", "about", "get", "do", "me", 
        "my", "our", "any", "your", "only", "some", "them", "other", "than", "then", 
        "now", "also", "into", "could", "these", "two", "its", "first", "over", 
        "because", "how", "after", "even", "most", "where", "made", "such", "did", 
        "many", "down", "should", "just", "those", "very", "much", "through", "before", 
        "good", "new", "well", "like", "make", "way", "see", "him", "know", "take", 
        "us", "come", "say", "use", "time", "year", "people", "said", "day", "being",
        "does", "too", "without", "don", "same", "while", "doesn", "didn", 
        "isn", "aren", "wasn", "weren", "haven", "hasn", "every", "again", "need", 
        "want", "look", "right", "call", "set", "show", "start", "last", "count", 
        "point", "used", "still", "here", "going", "between", "against", "high", 
        "help", "found", "great", "run", "find", "read", "real", "support", "really", 
        "fact", "both", "each", "says", "during", "few", "since", "turn", 
        "second", "something", "own", "may", "might", "must", "always", "never", 
        "another", "however", "almost", "often", "already", "though", "although", 
        "within", "whose", "whom", "doing", "done"
    }
    
    try:
        import nltk
        nltk.download('words', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        
        from nltk.corpus import words
        from nltk import pos_tag
        
        valid_words = set(w.lower() for w in words.words())
        print("✅ NLTK loaded successfully. Using NLTK Words corpus for strict filtering and POS tagging.")
    except Exception as e:
        print(f"❌ Failed to load NLTK data: {e}")
        return

    pure_words = []
    
    print("Filtering vocabulary...")
    for token, token_id in sorted_vocab:
        if not token.startswith("Ġ"): continue
        clean_word = token[1:].lower()
        
        if not re.match(r'^[a-z]+$', clean_word): continue
        if len(clean_word) <= 3: continue
        if clean_word in extended_stopwords: continue
        if clean_word not in valid_words: continue
            
        pure_words.append(token[1:])
        if len(pure_words) >= top_n: break
            
    word_dict = {}
    tagged_words = pos_tag(pure_words)
    
    for word, tag in tagged_words:
        if tag.startswith('N'): category = "Noun"
        elif tag.startswith('V'): category = "Verb"
        elif tag.startswith('J'): category = "Adjective"
        elif tag.startswith('R'): category = "Adverb"
        else: category = "Other"
        word_dict[word] = category

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(word_dict, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 Successfully saved {len(word_dict)} pure words and POS categories to {save_path}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract top pure words from a model's tokenizer.")
    parser.add_argument("--model_path", type=str, default="./fst_1_3B_local", help="Path or HuggingFace ID of the model")
    parser.add_argument("--save_path", type=str, default="top_200_words.json", help="Output JSON file name")
    parser.add_argument("--top_n", type=int, default=200, help="Number of top words to extract")
    
    args = parser.parse_args()
    
    extract_and_save_top_words(
        model_path=args.model_path,
        top_n=args.top_n,
        save_path=args.save_path
    )