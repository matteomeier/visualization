import pandas as pd
import glob

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
        tag_dict = entities.get('mentions', {})
        for item in tag_dict:
            mention = item.get('text', {})
            mentions.append(mention)
    return mentions