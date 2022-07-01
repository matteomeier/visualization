import glob
from preprocessing import get_politician_tweets
import pathlib
file_path = pathlib.Path(__file__).parent.resolve()

input_loc = file_path.joinpath('../../../raw_data/recorded-tweets/*.json')
input_files = glob.glob(str(input_loc))

accounts_loc = file_path.joinpath('../../../raw_data/followed-accounts.json')
accounts_file = glob.glob(str(accounts_loc))

politician_tweets = get_politician_tweets(input_files, accounts_file)
politician_tweets.to_json(file_path.joinpath('../data/politician_tweets.json'))