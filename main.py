import pandas as pd
from pathlib import Path
import torch.nn as nn
from sklearn.model_selection import train_test_split


DATA_FILE = Path('data.csv')
data_frame = None

def clean_data() -> pd.DataFrame:
    """
    This function gets the data and remove
    unecessary rows from it
    """
    data_frame = pd.read_csv(DATA_FILE)
    data_frame = data_frame[data_frame.isna().sum(axis = 1) <= 4]
    return data_frame

class NeuralNetwork(nn.Module):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data   
    
    def split_data(self):
        self.train_data, self.test_data = train_test_split(self._data, train_size=0.80, test_size=0.20, random_state=42)
        
    def run(self):
        print('here')

# so do we just run chatgpt (chatgpt api is the  way ) and nmake it sovle it????? --yes -- thats goofy -- that's cs
# w can import decision tree
# instead of decision tree cant we just use scikit to do th esame things as pytorch-Neural Network?? yea


    
if __name__ == '__main__':
    data = clean_data()
    model = NeuralNetwork(data)
    model.run()
