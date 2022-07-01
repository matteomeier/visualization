import glob
from preprocessing import get_features
import pathlib
import csv

file_path = pathlib.Path(__file__).parent.resolve()

input_loc = file_path.joinpath('../../../raw_data/recorded-tweets/*.json')
input_files = glob.glob(str(input_loc))

# german stopwords from https://github.com/solariz/german_stopwords.git
german_stopwords = []
with open(file_path.joinpath('../data/german_stopwords_full.txt')) as f:
    reader = csv.reader(f)
    for row in reader:
        german_stopwords.append(row[0])

bow, tfidf, features = get_features(input_files, german_stopwords)

with open(file_path.joinpath('../data/bow.csv'), 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    wr.writerows(bow)

with open(file_path.joinpath('../data/tfidf.csv'), 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    wr.writerows(tfidf)

with open(file_path.joinpath('../data/features.csv'), 'w', encoding='utf-8') as f:
    for item in features:
        f.write("%s\n" % item)