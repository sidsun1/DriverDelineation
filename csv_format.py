import pandas as pd

df = pd.read_csv('data.csv', delimiter = ',', engine = 'python')
df.to_csv('data.csv', index = False)

# IN git-bash: split -l 9000 --additional-suffix=.csv data.csv chunk_