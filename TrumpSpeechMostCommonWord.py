'''
Analysis on Trump Speeches - Group 1
Stanley Tantysco - 2201814670
Girindra Ado - 2201843506
Most Common Words from each Trump speeches
'''
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import os
import numpy as np

#Reading all text files
filesDict={}
files=[]
for dirname, _, filenames in os.walk('datasets/'):
    for filename in filenames:
        files.append(os.path.join(dirname,filename))
        filesDict[os.path.join(dirname,filename)]=open(os.path.join(dirname,filename),'r', errors='ignore').read()
        #print(os.path.join(dirname,filename))

#Main program
print("Here is speeches of Trump:")
for i in range(0,len(files)):
    print(i,': ',files[i])
fileIn=int(input('Please select number: '))
text = filesDict[files[fileIn]]
#Get a list of words using Spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
spacy_tokens=[token.lemma_ for token in doc if token.is_stop != True \
              and token.is_punct != True and token.is_alpha == True]

#Get 15 Most Common Words
word_freq_spacy = Counter(spacy_tokens)
common_words = [word[0] for word in word_freq_spacy.most_common(15)]
common_counts = [word[1] for word in word_freq_spacy.most_common(15)]

#Plotting histogram of Most Common Words
plt.figure(figsize=(12,8))
indexes=np.arange(len(common_words))
width=0.7
plt.bar(indexes, common_counts, width)
plt.xticks(indexes + 0.01, common_words)
plt.title('Most 15 Common Words from ' + files[fileIn])
plt.show()



