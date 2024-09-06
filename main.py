import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

df = pd.read_csv("/content/Reviews.csv")

df.head()

ax=df['Score'].value_counts().sort_index() \
.plot(kind='bar',title='Count of Reviews by Stars',figsize=(10,5))
ax.set_xlabel('Review Stars')
plt.show()

example=df['Text'][50]
print(example)

nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()

res = {}

for i,row in tqdm(df.iterrows(),total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders =vaders.reset_index().rename(columns={'index':'Id'})
vaders= vaders.merge(df,how='left')

vaders.head()

ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

fig, c = plt.subplots(1,3,figsize=(15,5))
sns.barplot(data=vaders,x='Score',y='pos',ax=c[0])
sns.barplot(data=vaders,x='Score',y='neu',ax=c[1])
sns.barplot(data=vaders,x='Score',y='neg',ax=c[2])
c[0].set_title('Positive')
c[1].set_title('Neutral')
c[2].set_title('Negative')
plt.show()
