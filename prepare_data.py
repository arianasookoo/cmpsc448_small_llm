"""
Step 1: download a text dataset from an online source.

Downloads the Sherlock Holmes books by Arthur Conan Doyle (public domain) from
Project Gutenberg, strips the Gutenberg license header/footer, and writes the
combined text to data.txt.
"""
import urllib.request

# use the Windows/macOS certificate store, so downloads work behind antivirus or proxy HTTPS scanning
# (pip install truststore)
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# Project Gutenberg ebook ids
BOOKS = {
    244: 'A Study in Scarlet',
    2097: 'The Sign of the Four',
    1661: 'The Adventures of Sherlock Holmes',
    834: 'The Memoirs of Sherlock Holmes',
    2852: 'The Hound of the Baskervilles',
    108: 'The Return of Sherlock Holmes',
}

def strip_gutenberg(text):
    # keep only the text between the "*** START OF" and "*** END OF" markers
    start = text.find('*** START OF')
    start = text.find('\n', start) + 1
    end = text.find('*** END OF')
    return text[start:end].strip()

parts = []
for book_id, title in BOOKS.items():
    url = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
    print(f'downloading {title} ...')
    raw = urllib.request.urlopen(url).read().decode('utf-8-sig')
    parts.append(strip_gutenberg(raw))

text = '\n\n'.join(parts)
text = text.replace('\r\n', '\n')
# replace curly quotes etc. with plain ASCII so the character vocabulary stays small
for a, b in {'‘': "'", '’': "'", '“': '"', '”': '"', '—': '--', '–': '-'}.items():
    text = text.replace(a, b)
text = ''.join(c for c in text if c == '\n' or 32 <= ord(c) < 127)

with open('data.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f'wrote data.txt: {len(text):,} characters, {len(set(text))} unique characters')
