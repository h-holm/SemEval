def load_lexicon(filename):
    wordlist = []
    with open(filename, encoding='latin-1') as infile:
        for line in infile:
            line = line.strip()
            if line and not line.startswith(';'):
                wordlist.append(line)
    return wordlist


positive = load_lexicon('lexicon/positive-words.txt')
negative = load_lexicon('lexicon/negative-words.txt')
