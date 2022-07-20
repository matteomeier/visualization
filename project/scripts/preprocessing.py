import pandas as pd
import glob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import json
import re
import numpy as np
from ast import literal_eval

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

    output = pd.DataFrame()

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
    
def get_edges_of_hashtags(input_files: glob.glob) -> pd.DataFrame:
    """Returns the weighted edges of the given hashtags.
    Args:
        input_files (glob.glob): Input file list with twitter chunks.
    Returns:
        pd.DataFrame: DataFrame with source hashtag, target hashtag and weight.
    """
    output = pd.DataFrame()

    # iterate through files
    for file in input_files:
        with open(file, 'r') as f:
            # read json to df
            df = pd.read_json(f)

            # apply function get hashtags
            df['tags'] = df['entities'].apply(extract_hashtags)

            # drop tweets with no hashtags
            df_true = []
            for ind in df.index:
                if(len(df['tags'][ind])==0):
                    df_true.append(False)
                else:
                    df_true.append(True)
            df = df[df_true]

            # lower all hashtags
            for ind in df.index:
                df['tags'][ind] = [hashtag.lower() for hashtag in df['tags'][ind]]

            # create edges for network (pair hashtags together)
            # e.g. tweet with three hashtags "#cdu", "#spd" and "#gruene" will be splitted in six entries:
            # -> #cdu #spd 1
            # -> #cdu #gruene 1
            # -> #spd #cdu 1
            # -> #spd #gruene 1
            # -> #gruene #cdu 1
            # -> #gruene #spd 1
            df['tags'] = df['tags'].apply(lambda x: str(x).lower())

            # create empty dataframe
            df_edges = pd.DataFrame(columns=['source', 'target'])

            # add each possible edge to the dataframe
            for ind in df.index:
                hashtags = literal_eval(df['tags'][ind])
                for source_hashtag in hashtags:
                    target_hashtags = hashtags.copy()
                    target_hashtags.remove(source_hashtag)                  
                    for target_hashtag in target_hashtags:                        
                        df_edges = df_edges.append({"source": source_hashtag, "target": target_hashtag}, ignore_index=True)                        

            # aggregate dfs and groupby hashtag (below called "source")
            # columns: source, target, weight
            df_edges[['source', 'target', 'weight']] = df_edges.groupby(['source', 'target'],as_index=False).size()        

            # drop 'nan' since dataframe retained original shape and filled it with 'nan' values
            df_edges = df_edges.dropna()

            # remove reversed entries
            # e.g.
            # 1. #cdu #spd 1
            # ...
            # 3. #spd #cdu 1 <- remove
            # source: https://stackoverflow.com/questions/71548402/pandas-drop-row-if-another-row-has-the-same-values-but-the-columns-are-switche
            df_cleaned = df_edges[~df_edges.apply(frozenset,axis=1).duplicated()]

            # save to output dataframe
            output = pd.concat([output, df_cleaned], axis=0, ignore_index=True)
        
    output = output.groupby(['source', 'target'], as_index=False).sum('weight')
    return output

def get_edges_of_mentions(input_files: glob.glob) -> pd.DataFrame:
    """Returns the weighted edges of the given mentions.
    Args:
        input_files (glob.glob): Input file list with twitter chunks.
    Returns:
        pd.DataFrame: DataFrame with source mention, target mention and weight.
    """
    output = pd.DataFrame()

    # iterate through files
    for file in input_files:
        with open(file, 'r') as f:
            # read json to df
            df = pd.read_json(f)

            # apply function get mentions
            df['mentions'] = df['entities'].apply(extract_mentions)

            # drop tweets with no mentions
            df_true = []
            for ind in df.index:
                if(len(df['mentions'][ind])==0):
                    df_true.append(False)
                else:
                    df_true.append(True)
            df = df[df_true]

            # lower all mentions
            for ind in df.index:
                df['mentions'][ind] = [mention.lower() for mention in df['mentions'][ind]]

            # create edges for network (pair mentions together)
            df['mentions'] = df['mentions'].apply(lambda x: str(x).lower())

            # create empty dataframe
            df_edges = pd.DataFrame(columns=['source', 'target'])

            # add each possible edge to the dataframe
            for ind in df.index:
                mentions = literal_eval(df['mentions'][ind])
                for source_mention in mentions:
                    target_mentions = mentions.copy()
                    target_mentions.remove(source_mention)                  
                    for target_mention in target_mentions:                        
                        df_edges = df_edges.append({"source": source_mention, "target": target_mention}, ignore_index=True)                        

            # aggregate dfs and groupby mentions (below called "source")
            # columns: source, target, weight
            df_edges[['source', 'target', 'weight']] = df_edges.groupby(['source', 'target'],as_index=False).size()        

            # drop 'nan' since dataframe retained original shape and filled it with 'nan' values
            df_edges = df_edges.dropna()

            # remove reversed entries
            # source: https://stackoverflow.com/questions/71548402/pandas-drop-row-if-another-row-has-the-same-values-but-the-columns-are-switche
            df_cleaned = df_edges[~df_edges.apply(frozenset,axis=1).duplicated()]

            # save to output dataframe
            output = pd.concat([output, df_cleaned], axis=0, ignore_index=True)
        
    output = output.groupby(['source', 'target'], as_index=False).sum('weight')
    return output

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

def get_hashtag_tweets(input_files: glob.glob, hashtag: str) -> pd.DataFrame:
    """Filters the tweets for the defined hashtags.
    Returns date, text, politician and party.
    Args:
        input_files (glob.glob): Input file list with twitter chunks.
        hashtag (str): Input string for defined hashtag.
    Returns:
        pd.DataFrame: DataFrame with date, text.
    """
    # get accounts
    text_output = []
    date_output = []

    # iterate through files
    for file in input_files:
        with open(file, 'r') as f:
            # read json to df
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
                    if text.lower().find(hashtag) == -1:
                        pass
                    else:
                        date = pd.to_datetime(tweet.get('created_at'))
                        text_output.append(text)
                        date_output.append(date)
                    
    output = {'text': text_output, 'date': date_output}
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