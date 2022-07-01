import pandas as pd
import glob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import json
import re
import numpy as np

def count_hashtags(input_files: glob.glob) -> pd.DataFrame:
    """Returns the hashtag counts for every day in timeframe.

    Args:
        input_files (glob.glob): Input file list with twitter chunks.

    Returns:
        pd.DataFrame: DataFrame with hashtag counts and day.
    """
    output = pd.DataFrame()

    # iterate through files
    for file in input_files:
        with open(file, 'r') as f:
            # read json to df
            df = pd.read_json(f)

            # apply function get hashtags
            df['tags'] = df['entities'].apply(extract_hashtags)
            
            # explode tags to rows and drop na values
            df = df.explode('tags')
            df = df[df['tags'].notna()]

            # change datetime to date and lower all hashtags
            df['created_at'] = pd.to_datetime(df['created_at']).dt.date
            df['tags'] = df['tags'].apply(lambda x: str(x).lower())

            # aggregate dfs and groupby date and hashtag
            df_agg = pd.DataFrame()
            df_agg[['date', 'hashtag', 'count']] = df.groupby(['created_at', 'tags'], as_index=False)['id'].count()

            # save to output dataframe
            output = pd.concat([output, df_agg], axis=0, ignore_index=True)
        
    output = output.groupby(['date', 'hashtag'], as_index=False).sum('count')
    return output

def extract_hashtags(entities: dict) -> list:
    """Extracts the hashtags from the entities field in each tweet.

    Args:
        entities (dict): Entities field from tweets.

    Returns:
        list: List of hashtags.
    """
    hashtags = []
    if type(entities)==dict:
        tag_dict = entities.get('hashtags', {})
        for item in tag_dict:
            tag = item.get('text', {})
            hashtags.append(tag)
    return hashtags

def count_mentions(input_files: glob.glob) -> pd.DataFrame:
    """Returns the mention counts for every day in timeframe.

    Args:
        input_files (glob.glob): Input file list with twitter chunks.

    Returns:
        pd.DataFrame: DataFrame with mention counts and day.
    """
    output = pd.DataFrame()

    # iterate through files
    for file in input_files:
        with open(file, 'r') as f:
            # read json to df
            df = pd.read_json(f)

            # apply function get mentions
            df['mentions'] = df['entities'].apply(extract_mentions)
            
            # explode mentions to rows and drop na values
            df = df.explode('mentions')
            df = df[df['mentions'].notna()]

            # change datetime to date
            df['created_at'] = pd.to_datetime(df['created_at']).dt.date

            # aggregate dfs and groupby date and hashtag
            df_agg = pd.DataFrame()
            df_agg[['date', 'mention', 'count']] = df.groupby(['created_at', 'mentions'], as_index=False)['id'].count()

            # save to output dataframe
            output = pd.concat([output, df_agg], axis=0, ignore_index=True)
        
    output = output.groupby(['date', 'mention'], as_index=False).sum('count')
    return output

def extract_mentions(entities: dict) -> list:
    """Extracts the mentions from the entities field in each tweet.

    Args:
        entities (dict): Entities field from tweets.

    Returns:
        list: List of mentions.
    """
    mentions = []
    if type(entities)==dict:
        tag_dict = entities.get('user_mentions', {})
        for item in tag_dict:
            mention = item.get('screen_name', {})
            mentions.append(mention)
    return mentions

def get_politician_tweets(input_files: glob.glob, accounts_file: glob.glob) -> pd.DataFrame:
    """Filters the tweets for the defined politicians accounts.
    Returns date, text, politician and party.

    Args:
        input_files (glob.glob): Input file list with twitter chunks.
        accounts_file (glob.glob): Input file for defined accounts.

    Returns:
        pd.DataFrame: DataFrame with date, text, politician and party.
    """
    # get accounts
    for file in accounts_file:
        with open(file, 'r', encoding='utf-8') as f:
            accounts = json.load(f)

    accounts_list = []
    for party in accounts:
        for account in accounts[party]:
            accounts_list.append(account)

    user_output = []
    text_output = []
    date_output = []
    party_output = []

    # iterate through files
    for file in input_files:
        with open(file, 'r') as f:
            # read json to df
            tweets = json.load(f)
            for tweet in tweets:
                if tweet.get('user') is not None:
                    user = tweet['user'].get('screen_name') 
                    if user in accounts_list:
                        # get only tweets which are no retweets
                        if tweet.get('retweeted_status') is None:
                        
                            if tweet.get('ext_tweet') is None or tweet.get('ext_tweet') == '':
                                if tweet.get('text') is None:
                                    text = ''
                                else:
                                    text = tweet['text']
                            else:
                                text = tweet['ext_tweet'].get('full_text')
                            date = pd.to_datetime(tweet.get('created_at'))
                            user_output.append(user)
                            text_output.append(text)
                            date_output.append(date)

    for i in range(len(user_output)):
        for party, account in accounts.items():
            if user_output[i] in account:
                party_output.append(party)
    output = {'user': user_output, 'text': text_output, 'date': date_output, 'party': party_output}
    output = pd.DataFrame(output)
    return output
    

def get_features(input_files: glob.glob, german_stopwords):
    """Extracts bag-of-words, tfidf features for tweets text.

    Args:
        input_files (glob.glob): Input file list with twitter chunks.

    Returns:
        list: bag-of-words.
        list: features.
        list: tfidf.
    """
    docs = []
    multipliers = []

    # iterate through files
    for file in input_files:
        with open(file, 'r') as f:
            tweets = json.load(f)
            for tweet in tweets:
                if tweet.get('retweeted_status') is None:
                    
                    if tweet.get('ext_tweet') is None or tweet.get('ext_tweet') == '':
                        if tweet.get('text') is None:
                            text = ''
                        else:
                            text = tweet['text']
                    else:
                        text = tweet['ext_tweet'].get('full_text')
                    
                    # remove mentions
                    text = re.sub('@[A-Za-z0-9_]+', '', text)
                    # remove hashtags
                    text = re.sub('#[A-Za-z0-9_]+', '', text)
                    # remove numbers
                    text = re.sub('[0-9]', '', text)
                    # remove links
                    text = re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)', '', text)
                    docs.append(text)

                    if tweet.get('retweet_count') is None:
                        multiplier = 1
                    else:
                        multiplier = tweet['retweet_count'] + 1

                    multipliers.append(multiplier)
                else:
                    pass

    bow_vectorizer = CountVectorizer(lowercase=True, stop_words=german_stopwords, analyzer='word', ngram_range=(2,3), max_features=250, max_df=0.8)
    tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True)
    bow = bow_vectorizer.fit_transform(docs).toarray()
    for i in range(len(bow)):
        bow[i] = np.multiply(bow[i], multipliers[i])
    features = bow_vectorizer.get_feature_names_out()
    tfidf = tfidf_transformer.fit_transform(bow).toarray()

    return bow, tfidf, features