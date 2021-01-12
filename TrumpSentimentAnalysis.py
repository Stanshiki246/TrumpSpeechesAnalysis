'''
Analysis on Trump Speeches - Group 1
Stanley Tantysco - 2201814670
Girindra Ado - 2201843506
Trump Speeches Multiple Data Visualizations
'''
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import itertools
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

#Get CountVectorizer of text function
def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

#Making histogram of Monthly Rallies function
def bar(data):
    width=0.7
    plt.bar(data.Month.value_counts().keys(), data.Month.value_counts().values, width)
    plt.ylabel('Numbers of Rallies')
    plt.title('Distribution Of Rallies Over Different Months')
    plt.show()

#Making pie of Yearly Rallies function
def pie(data):
    fig1,ax1 = plt.subplots()
    ax1.pie(x=data.Year.value_counts().values, labels=data.Year.value_counts().keys(), autopct='%1.1f%%')
    ax1.axis('equal')
    plt.legend()
    plt.title('Years of Rallies')
    plt.show()

#Making scatter and plotting of spread of sentiments vs number of sentences function
def scatter_and_plot(data):
    slope_neg, intercept_neg, r_value_neg, p_value_neg, std_err_neg = stats.linregress(data['# Of Sentences'],
                                                                                       (data['Negative Sentiments']-
                                                                                        data['Negative Sentiments'].mean())/
                                                                                       data['Negative Sentiments'].std())
    slope_pos, intercept_pos, r_value_pos, p_value_pos, std_err_pos = stats.linregress(data['# Of Sentences'],
                                                                                       (data['Positive Sentiments']-
                                                                                        data['Positive Sentiments'].mean())/
                                                                                       data['Positive Sentiments'].std())

    plt.scatter(x=data['# Of Sentences'],
                y=(data['Negative Sentiments']-data['Negative Sentiments'].mean())/data['Negative Sentiments'].std(),
                color='r', label='Negative Sentiments')
    plt.scatter(x=data['# Of Sentences'],
                y=(data['Positive Sentiments']-data['Positive Sentiments'].mean())/data['Positive Sentiments'].std(),
                color='b', label='Positive Sentiments')
    plt.plot(data['# Of Sentences'], slope_neg*data['# Of Sentences']+intercept_neg, label='Negative Trend')
    plt.plot(data['# Of Sentences'], slope_pos*data['# Of Sentences']+intercept_pos, label='Positive Trend')
    plt.legend()
    plt.title('Spread of Sentiments vs Number of Sentences in Trump Speeches')
    plt.ylabel('Z-Score')
    plt.xlabel('Number of Sentences')
    plt.show()

#Making confusion matrix of Trump speeches function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix of Trump Speeches', cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j],fmt),horizontalalignment='center',color='yellow' if cm[i,j] < thresh else 'red',
                 fontsize=40)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    plt.show()

#Making histogram of Most 15 Common Negative words in Trump Speeches function
def bar_most_negative_words(df):
    neg_word_list=[]
    stop_words = set(w.lower() for w in stopwords.words())
    sid = SentimentIntensityAnalyzer()
    text = ' '.join(df.Speech.values)
    tokenized_sentence = text.split(' ')
    tokenized_sentence = [w for w in tokenized_sentence if w not in stop_words]
    for word in tokenized_sentence:
        if (sid.polarity_scores(word)['compound']) <= -0.1:
            neg_word_list.append(word)
    neg_word_dict = Counter(neg_word_list)

    common_words = [word[0] for word in neg_word_dict.most_common(15)]
    common_counts = [word[1] for word in neg_word_dict.most_common(15)]

    plt.figure(figsize=(12,8))

    indexes=np.arange(len(common_words))
    width=0.7
    plt.bar(indexes, common_counts, width)
    plt.xticks(indexes + 0.01, common_words)
    plt.title('Most 15 Common Negative Words from Trump Speeches')
    plt.show()

#Making histogram of Most 15 Common Positive words in Trump Speeches function
def bar_most_positive_words(df):
    pos_word_list=[]
    stop_words = set(w.lower() for w in stopwords.words())
    sid = SentimentIntensityAnalyzer()
    text = ' '.join(df.Speech.values)
    tokenized_sentence = text.split(' ')
    tokenized_sentence = [w for w in tokenized_sentence if w not in stop_words]
    for word in tokenized_sentence:
        if (sid.polarity_scores(word)['compound']) >= 0.1:
            pos_word_list.append(word)
    pos_word_dict = Counter(pos_word_list)

    common_words = [word[0] for word in pos_word_dict.most_common(15)]
    common_counts = [word[1] for word in pos_word_dict.most_common(15)]

    plt.figure(figsize=(12,8))

    indexes=np.arange(len(common_words))
    width=0.7
    plt.bar(indexes, common_counts, width)
    plt.xticks(indexes + 0.01, common_words)
    plt.title('Most 15 Common Positive Words from Trump Speeches')
    plt.show()

def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

df=pd.read_csv('TrumpDatasets.csv')#Reading data from csv file
#print(df.info())

#Make lists for train and test split
list_corpus=df['Cleaned Text'].to_list()
list_sentiment=df['Sentiment Label'].to_list()
#Do train and test split
x_train,x_test,y_train,y_test = train_test_split(list_corpus,list_sentiment,test_size=0.2,random_state=40)
#Get counts of X train and test
x_train_counts, count_vectorizer = cv(x_train)
x_test_counts = count_vectorizer.transform(x_test)
#Get True Labels
clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial',
                         n_jobs= -1, random_state=40)
clf.fit(x_train_counts,y_train)
#Get Predicted labels
y_predicted_counts = clf.predict(x_test_counts)


'''

'''
#print(df['Sentiment Label'])

#Main program
print('Data Visualizations')
print('1. Distribution Of Rallies Over Different Months')
print('2. Years of Rallies')
print('3. Spread of Sentiments vs Number of Sentences in Trump Speeches')
print('4. Confusion Matrix of Trump Speeches')
print('5. Most 15 Common Negative Words from Trump Speeches')
print('6. Most 15 Common Positive Words from Trump Speeches')
choice=int(input('Select: '))
if choice == 1:
    bar(df)
elif choice == 2:
    pie(df)
elif choice == 3:
    scatter_and_plot(df)
elif choice == 4:
    cm = confusion_matrix(y_test,y_predicted_counts)
    plot_confusion_matrix(cm, classes=['pos','neu','neg'],normalize=False)
    accuracy, precision, recall, f1 = get_metrics(y_test,y_predicted_counts)
    print("accuracy = %.3f\nprecision = %.3f\nrecall = %.3f\nf1 = %.3f" % (accuracy, precision, recall, f1))
elif choice == 5:
    bar_most_negative_words(df)
elif choice == 6:
    bar_most_positive_words(df)
