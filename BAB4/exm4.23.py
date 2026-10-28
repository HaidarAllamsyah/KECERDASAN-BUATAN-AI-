# Example 4.23
# https://github.com/huggingface/transformers
# pip install transformers   (butuh juga: pip install torch   atau   tensorflow)
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
print(classifier('This is a good movie.'))      # [FIX] dibungkus print() agar tampil di skrip
