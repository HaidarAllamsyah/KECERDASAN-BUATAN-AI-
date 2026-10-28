# Exercise 4.20 - tanya jawab dengan teks lain
from transformers import pipeline

question_answerer = pipeline('question-answering')

context = ("Sinar Digital Ltd was founded in 2010 by Rina Wijaya in Surabaya. "
           "The company has 45 employees and builds mobile games.")

pertanyaan = [
    "Who founded the company?",
    "When was the company founded?",
    "Where is the company located?",
    "How many employees does the company have?",
    "What does the company build?",
    "What is the name of the CEO?",            # jawabannya TIDAK ada di teks
]

for q in pertanyaan:
    r = question_answerer(question=q, context=context)
    print(f"{q:45s} -> {r['answer']!r:22s} skor={r['score']:.3f}")
