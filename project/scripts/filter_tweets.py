import glob
from preprocessing import get_politician_tweets, get_hashtag_tweets
import pathlib
file_path = pathlib.Path(__file__).parent.resolve()

input_loc = file_path.joinpath('../../../raw_data/recorded-tweets/*.json')
input_files = glob.glob(str(input_loc))

accounts_loc = file_path.joinpath('../../../raw_data/followed-accounts.json')
accounts_file = glob.glob(str(accounts_loc))

politician_tweets = get_politician_tweets(input_files, accounts_file)
politician_tweets.to_json(file_path.joinpath('../data/politician_tweets.json'))

hashtag = '#afd'
hashtag_tweets = get_hashtag_tweets(input_files, hashtag)
hashtag_tweets.to_json(file_path.joinpath('../data/hashtag_tweets.json'))