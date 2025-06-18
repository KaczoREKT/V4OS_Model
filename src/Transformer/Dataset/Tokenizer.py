import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import xml.etree.ElementTree as ET

from tqdm import tqdm


def extract_wiki_text(xml_path, out_path):
    with open(xml_path, 'r', encoding='utf-8') as infile, open(out_path, 'w', encoding='utf-8') as outfile:
        # iterparse pozwala nie ładować całości do RAM
        context = ET.iterparse(infile, events=('end',))
        for event, elem in context:
            print(event, elem.tag)
            if elem.tag.endswith('page'):
                title = elem.find('./{*}title')
                text = elem.find('./{*}revision/{*}text')
                if text is not None and text.text:
                    # Opcjonalnie: filtruj brudne/nieduże/nieencyklopedyczne wpisy
                    article = text.text.replace('\n', ' ').replace('\r', ' ')
                    # Możesz odfiltrować niektóre rzeczy, np. dyskusje itp.
                    if article.strip() and len(article) > 100:  # pomiń bardzo krótkie wpisy
                        outfile.write(article.strip() + '\n')
                elem.clear()
    print(f"Zapisano czysty tekst do {out_path}")

# --- PROGRESS TOKENIZER ---
def wiki_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Czytam wiki.txt do tokenizera"):
            yield line.strip()


# 1. Utwórz tokenizer BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# 2. Ustaw trainer (specjalne tokeny – ważna kolejność!)
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    vocab_size=32000,          # możesz dać 16k/32k/50k
    min_frequency=2
)
# 3. Pre-tokenizer – dzielenie po spacji (Whitespace)
tokenizer.pre_tokenizer = Whitespace()
# 4. Wytrenuj na pliku tekstowym
if not os.path.exists('wiki.txt'):
    extract_wiki_text('D:/plwiki-latest-pages-articles.xml', 'wiki.txt')
files = ['wiki.txt']
tokenizer.train_from_iterator(wiki_lines('wiki.txt'), trainer=trainer)
tokenizer.save("polish-bpe-tokenizer.json")
print("Tokenizer zapisany jako polish-bpe-tokenizer.json!")
