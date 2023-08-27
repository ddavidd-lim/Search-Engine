import json

with open('inverted_index.json', 'r') as f:
    data = json.load(f)
    print(data)

# index = {token: [(doc_id, term_freq)]}