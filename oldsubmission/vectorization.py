#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import pickle
import os

if not os.path.isdir('model/'):
    os.makedirs('model/')

stop = stopwords.words('english')
stop.extend([x.replace("\'", "") for x in stop])
stop.extend(['nbsp', 'also', 'really', 'ive', 'even', 'jon', 'lot', 'could', 'many'])

path2 = "data/posts5monthscleaned.pkl"
path3 = "data/posts5monthsvectorized.pkl"
path4 = "data/postsbyauthor.pkl"
path5 = "data/postsbysubreddit.pkl"

original_df = pd.read_pickle(path2)
original_df['text'] = original_df.title + ' ' + original_df.selftext
original_df = original_df.drop(['title', 'selftext'], axis = 1)
user = pd.read_pickle(path4)
subreddit = pd.read_pickle(path5)

# Count vectorization for LDA
cv = CountVectorizer(token_pattern='\\w{3,}', max_df=.30, min_df=.0001, 
                     stop_words=stop, ngram_range=(1,1), lowercase=False,
                     dtype='uint8')
cv_fit = cv.fit_transform(original_df.text).transpose()
cvmodel = 'model/count_vec_model_full.pkl'
#pickle.dump(cv_fit, open(cvmodel, 'wb'))

cv_fit1 = cv.fit_transform(user.text).transpose()
cvmodel1 = 'model/cv_user.pkl'
#pickle.dump(cv_fit1, open(cvmodel1, 'wb'))

cv_fit2 = cv.fit_transform(subreddit.text).transpose()
cvmodel2 = 'model/cv_sub.pkl'
#pickle.dump(cv_fit2, open(cvmodel2, 'wb'))
