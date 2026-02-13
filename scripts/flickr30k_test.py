import pandas as pd

annotations = pd.read_table('../data/raw/flickr30k/results_20130124.token', sep='\t', header=None,
                            names=['image', 'caption'])
print(annotations)