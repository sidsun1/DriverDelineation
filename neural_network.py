import pandas as pd
from pathlib import Path
import torch as T
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_FILE = Path('data/data.csv')
data_frame = None

X = ['city_name', 'signup_os', 'signup_channel', 'vehicle_make', 'vehicle_model']
label_columns = ['completed']
learning_rate = 1
epochs = 50

def clean_data() -> pd.DataFrame:
    """
    Retrieves data from the csv and removes unecessary rows from it
    Cleans up rows with too many NA entries, and removes the id column, resetting the index
    """
    
    data_frame = pd.read_csv(DATA_FILE)
    # data_frame = data_frame[data_frame.isna().sum(axis = 1) <= 5] # Max 5 NA values

    # Shift the data frame over by one to drop the id number
    data_frame = data_frame.iloc[:, 1:]
    data_frame.reset_index(drop = True, inplace = True)

    # Fill na with values
    data_frame.replace("NA", pd.NA, inplace = True)

    date_cols = ['signup_date', 'bgc_date', 'vehicle_added_date', 'first_completed_date', 'vehicle_year']

    for col in date_cols:
        data_frame[col] = pd.to_datetime(data_frame[col], errors = 'coerce')

    data_frame['completed'] = data_frame['first_completed_date'].notna().astype(int)

    data_frame = data_frame[['city_name', 'signup_os', 'signup_channel', 'vehicle_make', 'vehicle_model', 'completed']]

    cat_cols = data_frame.select_dtypes(include=['object']).columns

    label_encoder = LabelEncoder()
    for col in cat_cols:
        if data_frame[col].dtype == 'object':
             data_frame[col] = label_encoder.fit_transform(data_frame[col].astype(str))

    print(f'Columns: {data_frame.columns}')

    return data_frame

class NeuralNetwork(T.nn.Module):
    def __init__(self, data, input_size:int, output_size:int):
        super(NeuralNetwork, self).__init__()

        self.lin1 = T.nn.Linear(input_size, 512)
        self.lin2 = T.nn.Linear(512, 256)
        self.lin3 = T.nn.Linear(256, 128)
        self.lin4 = T.nn.Linear(128, 64)
        self.lin5 = T.nn.Linear(64, 32)
        self.lin6 = T.nn.Linear(32, 16)
        self.lin7 = T.nn.Linear(16, 8)
        self.lin8 = T.nn.Linear(8, 4)
        self.lin9 = T.nn.Linear(4, 2)
        self.lin10 = T.nn.Linear(2, 1)
        self.lin11 = T.nn.Linear(1, 1)
        self.drop = T.nn.Dropout(.6)
        self.output = T.nn.Linear(1, output_size)
        self._data = data

    def forward(self, x):
        x = T.relu(self.lin1(x))
        x = T.relu(self.lin2(x))
        x = T.relu(self.lin3(x))
        x = T.relu(self.lin4(x))
        x = T.relu(self.lin5(x))
        x = T.relu(self.lin6(x))
        x = T.relu(self.lin7(x))
        x = T.relu(self.lin8(x))
        x = T.relu(self.lin9(x))
        x = T.relu(self.lin10(x))
        x = self.drop(x)
        x = T.relu(self.lin11(x))
        x = self.output(x)

        return x

    def split_data(self):
        self.train_data, self.test_data = train_test_split(self._data, train_size=0.95, test_size=0.05, random_state=70)

    def run(self):
        self.split_data()

        X_train = self.train_data[X].values.astype('float32')
        y_train = self.train_data[label_columns].values.astype('float32')
        X_test = self.test_data[X].values.astype('float32')
        y_test = self.test_data[label_columns].values.astype('float32')

        X_train = T.tensor(X_train)
        y_train = T.tensor(y_train)
        X_test = T.tensor(X_test)
        y_test = T.tensor(y_test)

        if T.cuda.is_available():
            device = T.device('cuda')
        else:
            device = T.device('cpu')

        self.to(device)
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        loss_fn = T.nn.BCEWithLogitsLoss()
        optimizer = T.optim.Adam(self.parameters(), lr=learning_rate)

        for e in range(epochs):
            print(f'Training epoch {e + 1}...')

            self.train()
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = loss_fn(outputs, y_train)
            loss.backward()
            optimizer.step()

            print(f'Epoch {e + 1} / {epochs} - Loss: {loss.item():.4f}')
        
        self.eval()
        with T.no_grad():
            predictions = self.forward(X_test)
            predicted_probs = T.sigmoid(predictions)
            predicted_classes = (predicted_probs > 0.5).float()
            correct = (predicted_classes == y_test).sum().item()
            total = len(y_test)
            acc = correct / total

            print(f'Accuracy {acc:.4f}')
            predictions_np = predicted_classes.cpu().numpy()
            y_test_np = y_test.cpu().numpy()

            self.plot_results(predictions_np, y_test_np)

    def plot_results(self, predictions, actual):
        correct = (predictions == actual).sum()
        wrong = (predictions != actual).sum()

        plt.figure(figsize=(6, 4))
        plt.bar(['Correct', 'Incorrect'], [correct, wrong], color=['blue', 'red'])
        plt.title('Model\'s Results - Correct vs Incorrect')
        plt.xlabel('Prediction Status')
        plt.ylabel('Count')
        plt.show()

if __name__ == '__main__':
    data = clean_data()
    model = NeuralNetwork(data, len(X), len(label_columns))
    model.run()
