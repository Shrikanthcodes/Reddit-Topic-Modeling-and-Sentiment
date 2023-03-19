-----------------------------------------------------------------
FILES TO OPEN:
-----------------------------------------------------------------

topicModel_and_sentiment.ipynb

visual.html for interactive output.

*There are no other dependancies in the code. To view the previously submitted segmented submission, please view the /oldsubmission folder

Challenges Faced:
-----------------------------------------------------------------
1) Data Scaling; Used data for the month of Oct 2018, instead of using data over 6 months as planned, due to time consideration and compute power throttling issues

2) Time and deadline constraints; LDA model could have been tuned further

Code Source:
------------------------------------------------------------------
In topicModel_and_sentiment.ipynb, the code for pyLDAvis is from the official documentation, and not completely written by me.

Data Source:
-------------------------------------------------------------------
Data is imported via an API call to google BigQuery repository, as demonstrated in the code.

Data Preview:
-------------------------------------------------------------------
Data columns:created_utc, subreddit, author, domain, url, num_comments, score, title, selftext, id, gilded, retrieved_on, over_18.

Lines of code
------------------------------------------------------------------------
2969 lines in total (according to submission stats in github), approx 400 lines of actual code: in topicModel_and_sentiment.ipynb

Bibliography
------------------------------------------------------------------------
1) Cable cord sentiment analysis: https://github.com/akhilesh-reddy/Cable-cord-cutter-Sentiment-analysis-using-Reddit-data#introduction

2) Sentiment analysis using product review data By Xing Fang, Justin Zhan:

Credits:
------------------------------------------------------------------------
Code By Shrikanth Subramanian, for Applied Data Analysis class

Report:
------------------------------------------------------------------------
Key Terms:

Subreddit: A reddit forum where like minded individuals congregate to discuss their mutual interests.

Topic Modeling: Topic modeling is an unsupervised ML technique that is capable of scanning a text, detecting patterns within them, and clustering word groups and similar expressions that best characterize a set of documents.

Sentiment Analysis: The mechanism of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic

Introduction:

In this project, I analyzed all posts on reddit from the month of October 2018. In order to complete this project, I leveraged the tools provided by google cloud platform (GCP) suite. My objective through this project was to cluster similar subreddits together and to find interesting insights from the correlation between interest in a particular topic, the sentiment score corresponding to it, and the likelihood of that subreddit being banned in the future. I used concepts such as LDA (Latent Dirichlet Allocation), Sentiment Analysis, Tf-IDF and count vectorization to analyze textual data from approximately 175,000 Reddit users' posts. This large amount of data helped me classify and tag different user groups. This could be further developed in the future to help law enforcement and the reddit censorship team weed out problematic subreddits before they can cause too much havoc. 
This grouping of people, and subreddits based on similarity and likemindedness could also contribute to an advertising revenue growth increase. This could serve as an interactive outlook on data and classification which, if scaled up, could lead to some very interesting inferences. For topic modeling, which is the major task performed in this project, I used LDA after trying similarity index, and SVM as alternatives. The reason for this is that it uses information from multiple features to create a new axis in turn minimizing the standard deviation and maximizes the class distance of the two variables

Data Acquisition/ Description/ Preprocessing:

Using the publicly available fh-bigquery reddit_posts dataset, I queried the October 2018 subset for all self-posts with at least 500 characters and a score of greater than +15 or less than -15 to filter for good/bad content while still retaining a good sample size of data for both cases. Many subreddits were filtered out due to duplicates, bad noise, foreign languages, and bot behavior. Bot behavior was especially tagged  by a subreddit called r/removalbot which holds a list of most active registered bots. These were my initial biases to the data and my analyses and conclusions drawn are biased, they are not meant to be a representative view of all of Reddit and conclusions only pertain to the approximately 174,000 posts I queried.
The columns returned: created_utc, subreddit, author, domain, url, num_comments, score, title, selftext, id, gilded, retrieved_on, over_18.
During cleaning I removed links, html character entities (i.e. &amp;), newline breaks, punctuation, and put all characters in lowercase.
Using nltk's english stopwords, I was able to tokenize and eliminate posts with very few characters, I was also able to use a count vectorization method to vectorize the data to form better insights (this worked better than tf-idf method, as demonstrated in /oldsubmission/vectorization.py, I found this to be more useful than the tf-idf model because of the nature and complexity of the data. CountVectorizer performed the task of tokenizing and counting. I did not use n-grams method. I used gensim corpus method for creating a corpus from a sparse matrix, before I could run a LDA classifier.
I also performed a similarity index to compare the performance of the LDA with, but it returned worse results.
For the sentiment analysis, I used VaderSentiment python library which has pretrained models useful for analysis. I performed lemmatization using wordnet, and used the SentimentIntensityAnalyzer() function for calculating sentiment scores for each sentence. I then found the average result for the sample of 100 posts, and the normalized sentiment score was 0.47, which is very good considering -1 was a very bad score, and +1 was a very good sentiment score. 

Datasource:

fh-bigquery reddit_posts
Google BigQuery
Pre saved pickle (.pkl) files

Algorithms:

Latent Dirichlet Allocation for Topic Modeling
Count Vectorization
Sentiment analysis using Vader
Similarity index

Results and Insights:

Contrary to what one would assume, the random sampling of the data returned that the average sentiment score of reddit posts is 0.5, which is very good.
A Perplexity score of -9.146 and a coherence score of 0.8 proves that our LDA model is very well trained
Our verification of data also proved that the topic modeling was successful.

Future Work:

If I were to be able to further dig into the data, I would try to find out if there is a tangible correlation between text data (posts, comments) with a subreddit’s likelihood to get banned. While I was able to collect the data for this part of the project, due to time constraints, I had to give up on it. There are also many more methods that I would love to sample instead of LDA to get even better results. I originally wanted to use the score and no of comments columns as well, but I could only used title and selftext columns to perform topic modeling due to time constraints.

Project is created with:
Python 3.8; Packages: Nltk, gensim, Vader, pyLDAvis, Pickle
