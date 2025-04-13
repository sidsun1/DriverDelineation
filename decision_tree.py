import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier


class DriverDecisionModel:
    def __init__(self, data_path=Path('data/data.csv')):
        self.data_path = Path(data_path)
        self.df = None
        self.clf = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20, min_samples_leaf=10)
        self.features = [
            'city_name', 'signup_os', 'signup_channel', 'vehicle_make', 'vehicle_model', 'vehicle_year',
            'signup_date', 'bgc_date', 'vehicle_added_date'
        ]
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.y_pred = None

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df['first_completed'] = self.df['first_completed_date'].notna().astype(int)

        # Convert date columns to datetime then to ordinal (number of days)
        date_cols = ['signup_date', 'bgc_date', 'vehicle_added_date']
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            self.df[col] = self.df[col].map(lambda x: x.toordinal() if pd.notna(x) else 0)

        self.df['vehicle_year'] = self.df['vehicle_year'].fillna(0)

        # Encode categorical columns
        categorical_cols = ['city_name', 'signup_os', 'signup_channel', 'vehicle_make', 'vehicle_model']
        for col in categorical_cols:
            self.df[col] = pd.factorize(self.df[col])[0]

        # Drop only the unused columns
        self.df = self.df.drop(columns=['id', 'first_completed_date'])

    def split_data(self, test_size=10000):
        X = self.df[self.features]
        y = self.df['first_completed']
        self.X_train = X[:-test_size]
        self.y_train = y[:-test_size]
        self.X_test = X[-test_size:]
        self.y_test = y[-test_size:]

    def train_model(self):
        self.clf.fit(self.X_train, self.y_train)

    def evaluate_and_save_predictions(self, output_file=Path('data/predictions.csv')):
        self.y_pred = self.clf.predict(self.X_test)

        output_rows = []
        correct = 0
        for i in range(len(self.y_test)):
            predicted = self.y_pred[i]
            actual = self.y_test.iloc[i]
            if predicted == actual:
                correct += 1
            row_index = len(self.X_train) + i
            output_rows.append({
                'Row': row_index,
                'Predicted': predicted,
                'Actual': actual
            })
            print(f"Row {row_index}: Predicted = {predicted}, Actual = {actual}")

        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(output_file, index=False)

        accuracy = (correct / len(self.y_test)) * 100
        print(f"\nManual Accuracy: {accuracy:.2f}% ({correct} out of {len(self.y_test)} correct)")

    def visualize_predictions(self, input_file=Path('data/predictions.csv')):
        df = pd.read_csv(input_file)
        df['error'] = df['Predicted'] != df['Actual']

        correct = (df['error'] == False).sum()
        incorrect = df['error'].sum()

        plt.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'])
        plt.title('Model Prediction Performance')
        plt.ylabel('Number of Predictions')
        plt.show()

    def show_feature_importance(self):
        importances = self.clf.feature_importances_
        plt.figure(figsize=(8, 5))
        plt.bar(self.features, importances, color='blue')
        plt.title('Feature Importances')
        plt.ylabel('Importance Score')
        plt.xlabel('Feature')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run(self):
        self.load_and_prepare_data()
        self.split_data()
        self.train_model()
        self.evaluate_and_save_predictions()
        self.visualize_predictions()
        self.show_feature_importance()


if __name__ == '__main__':
    model = DriverDecisionModel()
    model.run()