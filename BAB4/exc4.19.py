# Exercise 4.19 - sentimen untuk beberapa kalimat
from transformers import pipeline

classifier = pipeline('sentiment-analysis')

kalimat = [
    "I love this phone, the camera is amazing.",          # jelas positif
    "This is the worst service I have ever had.",         # jelas negatif
    "The movie was not bad at all.",                      # negasi
    "Great, another delay. Just what I needed.",          # sarkasme
    "The package arrived on Tuesday.",                    # netral
    "Saya sangat suka film ini.",                         # bahasa Indonesia
]

for k, hasil in zip(kalimat, classifier(kalimat)):
    print(f"{hasil['label']:9s} {hasil['score']:.4f}   {k}")
