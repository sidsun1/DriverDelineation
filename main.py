import pandas as pd
from pathlib import Path
import torch as T
from sklearn.model_selection import train_test_split


DATA_FILE = Path('data.csv')
data_frame = None

def clean_data() -> pd.DataFrame:
    """
    Retrieves data from the csv and removes unecessary rows from it
    Cleans up rows with too many NA entries, and removes the id column, resetting the index
    """
    data_frame = pd.read_csv(DATA_FILE)
    data_frame = data_frame[data_frame.isna().sum(axis = 1) <= 4] # Max 4 NA values

    # Shift the data frame over by one to drop the id number
    data_frame = data_frame.iloc[:, 1:]
    data_frame.reset_index(drop = True, inplace = True)

    return data_frame

X = ['0', '4', '5', '6', '7', '8', '9']
label_columns = ['10', '11', '12', '13']

class NeuralNetwork(T.nn.Module):
    def __init__(self, data, input_size:int, output_size:int):
        super(NeuralNetwork, self).__init__()
        self._data = data
        self.lin1 = T.nn.Linear(input_size, 4600)
        self.lin2 = T.nn.Linear(4600, 5600)
        self.lin3 = T.nn.Linear(5600, 6600)
        self.lin4 = T.nn.Linear(6600, 7600)
        self.lin5 = T.nn.Linear(7600, 8600)
        self.lin6 = T.nn.Linear(8600, 9600)
        self.lin7 = T.nn.Linear(9600, 11600)
        self.lin8 = T.nn.Linear(11600, 12600)
        self.drop = T.nn.Dropout(0.1)
        self.output = T.nn.Linear(12600, output_size)
    
    def forward(self, x):
        x = T.tanh(self.lin1(x))
        x = T.sigmoid(self.lin2(x))
        x = T.tanh(self.lin3(x))
        x = T.sigmoid(self.lin4(x))
        x = self.drop(x)
        x = T.tanh(self.lin5(x))
        x = T.sigmoid(self.lin6(x))
        x = T.tanh(self.lin7(x))
        x = T.sigmoid(self.lin8(x))
        x = self.output(x)
        return x

    @property
    def data(self):
        return self._data   
    
    def split_data(self):
        self.train_data, self.test_data = train_test_split(self._data, train_size=0.80, test_size=0.20, random_state=42)
        
    def run(self):
        self.split_data()
        print(self.train_data)

if __name__ == '__main__':
    data = clean_data()
    model = NeuralNetwork(data)
    model.run()
