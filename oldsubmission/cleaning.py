import pandas as pd
import re
import nltk.corpus
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pickle

path = "data/posts5months.pkl"
path2 = "data/posts5monthscleaned.pkl"
#path3 = "data/posts5monthsvectorized.pkl"
path4 = "data/postsbyauthor.pkl"
path5 = "data/postsbysubreddit.pkl"
#model = 'count_vec_model.sav'
#pickle.dump(model, open(filename, 'wb'))

nltk.download('stopwords')
stop = stopwords.words('english')
stop.extend([x.replace("\'", "") for x in stop])
stop.extend(['nbsp', 'also', 'really', 'ive', 'even', 'jon', 'lot', 'could', 'many'])


original_df = pd.read_pickle(path)
'''
original_df = original_df.drop(['domain', 'url', 'gilded', 'retrieved_on'], axis = 1)
original_df = original_df.reset_index().rename(columns={"index": "position"})
original_df.to_pickle(path)
'''

def  clean_text(df, text):
    df[text] = df[text].str.lower()
    df[text] = df[text].apply(lambda elem: 
    re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem)) 
    df[text] = df[text].apply(lambda x:
    ' '.join([word for word in x.split() if word not in (stop)]))
    df[text] = df[text].str.replace("\\n", " ")
    df[text] = df[text].str.replace(r"http\S+", " ")
    df[text] = df[text].str.replace(r"#f", "")
    df[text] = df[text].str.replace(r"[\’\'\`\":]", "")
    # remove numbers
    df[text] = df[text].apply(lambda elem: 
    re.sub(r"\d+", "", elem))
    return df

df = clean_text(original_df, 'title')
df = clean_text(original_df, 'selftext')
#original_df.to_pickle(path2)

text_posts = original_df.loc[:,['title', 'selftext']]
original_df['text'] = text_posts.title + ' ' + text_posts.selftext
text_posts['text'] = text_posts.title + ' ' + text_posts.selftext
original_df = original_df.drop(['title', 'selftext'], axis = 1)
'''
print(original_df.head())
print(text_posts.head())
'''

def groupPosts(x):
    ''' Group users' id's by post '''
    return pd.Series(dict(id = ", ".join(x['id']),                    
                          text = ", ".join(x['text'])))


xf = original_df.groupby('subreddit').apply(groupPosts) 
df = original_df.groupby('author').apply(groupPosts) 
#xf.to_pickle(path5)
#df.to_pickle(path4)


print(df.head())
print(xf.head())

