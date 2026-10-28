# Exercise 4.18 - analisis NLTK dengan teks lain
import nltk
for r in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
          'averaged_perceptron_tagger_eng', 'maxent_ne_chunker',
          'maxent_ne_chunker_tab', 'words']:
    nltk.download(r, quiet=True)

teks = ("Elon Musk founded SpaceX in California in 2002. Tim Cook leads Apple, "
        "and Google has offices in Jakarta, Indonesia. "
        "Politeknik Elektronika Negeri Surabaya is a university in Surabaya.")

tokens = nltk.word_tokenize(teks)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)

print("Jumlah token:", len(tokens))
print(tagged)
print("Entitas:")
for subtree in entities:
    if hasattr(subtree, 'label'):
        print("  ", subtree.label(), "->", ' '.join(w for w, _ in subtree))
