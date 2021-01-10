'''
Analysis on Trump Speeches - Group 1
Stanley Tantysco - 2201814670
Girindra Ado - 2201843506
Convert all Trump speeches into a csv file
'''
import spacy
import pandas as pd
import os
import re
from nltk.corpus import words as english_words, stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from SpacyMagic import clean_text

#Reading all text files
filesDict={}
files=[]
for dirname, _, filenames in os.walk('datasets/'):
    for filename in filenames:
        files.append(os.path.join(dirname,filename))
        filesDict[os.path.join(dirname,filename)]=open(os.path.join(dirname,filename),'r', errors='ignore').read()

#Making dataset from text files
f_names = [f.replace('.txt','') for f in files]
months = ['Jan','Feb','Mar','Apr','May','Jul','Jun','Aug','Sep','Oct','Nov','Dec']
city, r_month, date = [],[],[]
for name in f_names:
    index = -1
    for month in months:
        index = name.find(month)
        if index != -1:
            r_month.append(month)
            break
    city.append(name[:index])
    date.append(name[index+3:])

df = pd.DataFrame({'Month': r_month, 'Year': date, 'City': city, 'Speech': filesDict.values(),'Text': filesDict.values()})
df['Day']= df['Year'].apply(lambda x: x.split('_')[0])
df['Year']= df['Year'].apply(lambda x: x.split('_')[1])
df = df[['Day','Month','Year','City','Speech','Text']]
df['City']= df['City'].apply(lambda x: ' '.join(re.sub(r"([A-Z])", r"\1", x).split()))
df['Speech']=df['Speech'].apply(lambda x: x.strip().lower())

for i in df.index:
    df.loc[i, 'Cleaned Text'] = clean_text(df.loc[i,'Text'])
df.drop(columns=['Text'],inplace=True)

sid = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")
stop_words = set(w.lower() for w in stopwords.words())

df['# Of Words']=df['Speech'].apply(lambda x: len(x.split(' ')))
df['# Of StopWords']=df['Speech'].apply(lambda x: len([word for word in x.split(' ') if word in stop_words]))
df['# Of Sentences']=df['Speech'].apply(lambda x: len(re.findall('\.', x)))
df['Average Word Length']=df['Speech'].apply(lambda x: np.mean(np.array([len(token) for token in x.split(' ')])))
df['Average Sentence Length']=df['Speech'].apply(lambda x: np.mean(np.array([len(token) for token in x.split('.')])))
df['Speech']=df['Speech'].apply(lambda x: re.sub(r'[,.;@#?!&$]+',' ',x))

df['Sentiments']=df['Speech'].apply(lambda x: sid.polarity_scores(x))
df['Positive Sentiments']=df['Sentiments'].apply(lambda x: x['pos'])
df['Neutral Sentiments']=df['Sentiments'].apply(lambda x: x['neu'])
df['Negative Sentiments']=df['Sentiments'].apply(lambda x: x['neg'])

df['Compound Value']=df['Sentiments'].apply(lambda x: x['compound'])
df.drop(columns=['Sentiments'],inplace=True)
compound=0
for i in df.index:
    compound=df.loc[i, 'Compound Value']
    if compound >= 1:
        df.loc[i, 'Sentiment Label'] = 'pos'
    elif compound <= -1:
        df.loc[i, 'Sentiment Label'] = 'neg'
    else:
        df.loc[i, 'Sentiment Label'] = 'neu'
df.drop(columns=['Compound Value'],inplace=True)

df['State'] = ['Mississippi','Minnesota','Ohio','Michigan','New Hampshire','Nevada','New Jersey','Texas','Iowa','North Carolina','Michigan','Kentucky','Wisconsin','Arizona',
                       'Oklahoma','Minnesota','New Hampshire','Pennsylvania','Colorado','Pennsylvania','Ohio','South Carolina','North Carolina','Nevada','North Carolina','New Hampshire',
                       'North Carolina','Ohio','Texas','Wisconsin','Nevada','South Carolina','New Mexico','Arizona','Pennsylvania']

df['# Of Different Countries Mentioned'] = df['Speech'].apply(lambda x: len([token for token in nlp(x).ents
                                                                             if token.label_ == 'GPE']))
df['# Of Different People Mentioned'] = df['Speech'].apply(lambda x: len([token for token in nlp(x).ents
                                                                             if token.label_ == 'PERSON']))

#print(df.info())
#Convert dataset into csv file
pd.DataFrame.to_csv(df,'TrumpDatasets.csv')
print('Saved Successfully')
