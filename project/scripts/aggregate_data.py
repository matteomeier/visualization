import glob
from preprocessing import count_hashtags, count_mentions
import pathlib
file_path = pathlib.Path(__file__).parent.resolve()

input_loc = file_path.joinpath('../../../raw_data/recorded-tweets/*.json')
input_files = glob.glob(str(input_loc))

hashtags_count = count_hashtags(input_files)
hashtags_count.to_json(file_path.joinpath('../data/hashtags_count.json'))

mentions_count = count_mentions(input_files)
mentions_count.to_json(file_path.joinpath('../data/mentions_count.json'))