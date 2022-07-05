import glob
from preprocessing import get_edges_of_hashtags, get_edges_of_mentions
import pathlib
file_path = pathlib.Path(__file__).parent.resolve()

input_loc = file_path.joinpath('../raw_data/recorded-tweets/*.json')
input_files = glob.glob(str(input_loc))

hashtags_weighted_edges = get_edges_of_hashtags(input_files)
hashtags_weighted_edges.to_json(file_path.joinpath('../data/hashtags_weighted_edges.json'))

mentions_weighted_edges = get_edges_of_mentions(input_files)
mentions_weighted_edges.to_json(file_path.joinpath('../data/mentions_weighted_edges.json'))