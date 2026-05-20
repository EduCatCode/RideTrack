import time
import math
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix
from .ppm import ppm

class DrivePSTs:

    def Introduction(self):
        """
        Function: Introduction to how to use this class and its methods.

        Note: This function will print out the guide to console.
        """
        intro = """

       ╔═╗ ╔╦═══╦╗ ╔╦╗ ╔╗╔═══╦═══╗╔╗╔═╦═══╦═══╦═══╦═══╗╔╗
       ║║╚╗║║╔═╗║║ ║║║ ║║║╔══╣╔══╝║║║╔╩╗╔╗╠╗╔╗║╔═╗║╔═╗╠╝║
       ║╔╗╚╝║║ ╚╣╚═╝║║ ║║║╚══╣╚══╗║╚╝╝ ║║║║║║║╠╝╔╝╠╝╔╝╠╗║
       ║║╚╗║║║ ╔╣╔═╗║║ ║║║╔══╣╔══╝║╔╗║ ║║║║║║║║ ║╔╬═╝╔╝║║
       ║║ ║║║╚═╝║║ ║║╚═╝║║╚══╣╚══╗║║║╚╦╝╚╝╠╝╚╝║ ║║║║╚═╦╝╚╗
       ╚╝ ╚═╩═══╩╝ ╚╩═══╝╚═══╩═══╝╚╝╚═╩═══╩═══╝ ╚╝╚═══╩══╝
                  ╔═══╗  ╔╗  ╔════╗       ╔╗
                  ║╔═╗║  ║║  ║╔╗╔╗║       ║║
                  ║╚═╝╠╦═╝╠══╬╝║║╚╬═╦══╦══╣║╔╗
                  ║╔╗╔╬╣╔╗║║═╣ ║║ ║╔╣╔╗║╔═╣╚╝╝
                  ║║║╚╣║╚╝║║═╣ ║║ ║║║╔╗║╚═╣╔╗╗
                  ╚╝╚═╩╩══╩══╝ ╚╝ ╚╝╚╝╚╩══╩╝╚╝
                ╭━━━╮        ╭━━━┳━━━┳━━━━╮
                ╰╮╭╮┃        ┃╭━╮┃╭━╮┃╭╮╭╮┃
                 ┃┃┃┣━┳┳╮╭┳━━┫╰━╯┃╰━━╋╯┃┃┣┻━╮
                 ┃┃┃┃╭╋┫╰╯┃┃━┫╭━━┻━━╮┃ ┃┃┃━━┫
                ╭╯╰╯┃┃┃┣╮╭┫┃━┫┃  ┃╰━╯┃ ┃┃┣━━┃
                ╰━━━┻╯╰╯╰╯╰━━┻╯  ╰━━━╯ ╰╯╰━━╯

        歡迎使用 RideTrack DrivePSTs 功能！
        
        這個Class包含以下功能：
        
        1. train_vomm: VoMM/PST模型訓練(可在加動作)。
           用法：train_vomm(self, train_data, l, k, save_model=None)
        
        2. test_vomm: 使用訓練完模型預測(含簡易過濾噪聲)。
           用法：test_vomm(self, data_set, frequency, save_path=None)
        
        3. compute_accuracy: 計算各駕駛行為準確度。
           用法：compute_accuracy(self, dataset, frequency, save_path=None):

        4. 用法：calculate_action_prediction_counts: 各駕駛行為混淆矩陣。
           用法：calculate_action_prediction_counts(self, test_label, test_predict, draw_plot=False)
        
        
        """
        print(intro)


    # 儲存檔案使用
    @staticmethod
    def _save_dataframe(df, path):
        """
        Function: Save dataframe to csv.

        Parameters:
            df: The dataframe to be saved.
            path: The path to save the dataframe.
        """
        df.to_csv(path, index=False)

    # 計算執行時間
    @staticmethod
    def _print_execution_time(start_time):
        """
        Function: Print the execution time from start_time to now.

        Parameters:
            start_time: The start time of execution.
        """
        # Compute and print the execution time
        execution_time = time.time() - start_time
        hours, rem = divmod(execution_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Execution time: {hours} hours {minutes} minutes {seconds} seconds")

        
    def load_vlmm_models(self, model_file):
        with open(model_file, 'rb') as f:
            self.models = pickle.load(f)

    def train_vomm(self, train_data, l, k, save_model=None):
        start_time = time.time()
        actions = ['Go Straight', 'Idle', 'Turn Left', 'Turn Right', 'Two-Stage Left', 'U-turn']

        self.models = []
        for action in actions:
            data = train_data.loc[train_data['Action'] == action, 'Action Element'].astype(int).tolist()
            model = ppm()
            model.fit(data, d=l, alphabet_size=k)
            self.models.append(model)

        if save_model:
            with open(save_model, 'wb') as f:
                pickle.dump(self.models, f)

        self._print_execution_time(start_time)

    def test_vomm(self, data_set, frequency, save_path=None):
        action_element_list = data_set['Action Element'].values.tolist()
        start_time = time.time()
        predictions = []
        actions = ['Go Straight', 'Idle','Turn Left', 'Turn Right', 'Two-Stage Left', 'U-turn']

        for num in tqdm(range(len(action_element_list))):
            max_score = float('-inf')
            selected_model = None

            for model, action in zip(self.models, actions):
                # Avoid division by zero if i is 0, though range starts from 6
                scores = [math.exp(model.logpdf(action_element_list[max(0, num-i):num])) ** (1/i) for i in range(6, 30)]
                model_score = max(scores)

                if model_score > max_score:
                    max_score = model_score
                    selected_model = action

            predictions.append(selected_model)

        data_set['Predict'] = predictions
        data_set['Filter_Predict'] = self.filter_actions(data_set['Predict'], frequency)

        if save_path:
            self._save_dataframe(data_set, save_path)

        self._print_execution_time(start_time)
        return data_set

    def train_vomm_transition(self, train_data, l, k, save_model=None):
        start_time = time.time()
        actions = ['Go Straight', 'Idle', 'Turn Left', 'Turn Right', 'Two-Stage Left', 'U-turn', 'Transition']

        self.models = []
        for action in actions:
            data = train_data.loc[train_data['Action'] == action, 'Action Element'].astype(int).tolist()
            model = ppm()
            model.fit(data, d=l, alphabet_size=k)
            self.models.append(model)

        if save_model:
            with open(save_model, 'wb') as f:
                pickle.dump(self.models, f)

        self._print_execution_time(start_time)

    def test_vomm_transition(self, data_set, frequency, save_path=None):
        action_element_list = data_set['Action Element'].values.tolist()
        start_time = time.time()
        predictions = []
        actions = ['Go Straight', 'Idle','Turn Left', 'Turn Right', 'Two-Stage Left', 'U-turn', 'Transition']

        for num in tqdm(range(len(action_element_list))):
            max_score = float('-inf')
            selected_model = None

            for model, action in zip(self.models, actions):
                scores = [math.exp(model.logpdf(action_element_list[max(0, num-i):num])) ** (1/i) for i in range(6, 30)]
                model_score = max(scores)

                if model_score > max_score:
                    max_score = model_score
                    selected_model = action

            predictions.append(selected_model)

        data_set['Predict'] = predictions
        data_set['Filter_Predict'] = self.filter_actions(data_set['Predict'], frequency)

        if save_path:
            self._save_dataframe(data_set, save_path)

        self._print_execution_time(start_time)
        return data_set
    
    @staticmethod
    def _mode_of_data(window):
        if not window:
            return None
        count = Counter(window)
        return count.most_common(1)[0][0]


        
    def filter_actions(self, dataset, frequency):
        filtered_data = []
        previous_action = None
        for i, action in enumerate(dataset):
            if action != previous_action:
                window = dataset[i:i+(2*frequency)]
                if self._mode_of_data(window) == action:
                    pass
                else:
                    if self._mode_of_data(window + dataset[i-frequency:i+frequency]) != action:
                        action = self._mode_of_data(window)
                    else:
                        action = previous_action
            filtered_data.append(action)
            previous_action = action
        return filtered_data

    def compute_accuracy(self, dataset, frequency, save_path=None):
        dataset['Filter_Predict'] = self.filter_actions(dataset['Predict'], frequency)
        filtered_data = dataset[['Action', 'Predict', 'Filter_Predict']].dropna()

        result_df1 = self._compute_total_accuracy(filtered_data, ['Predict', 'Filter_Predict'])
        result_df2 = self._compute_label_accuracy(filtered_data, ['Predict', 'Filter_Predict'])
        result_df2['Accuracy (Total)'] = result_df1['Accuracy (Total)']

        if save_path:
            self._save_dataframe(result_df2, save_path)    

        return result_df2

    @staticmethod
    def _compute_total_accuracy(data, columns):
        match_percentages = []
        for column in columns:
            count = (data['Action'] == data[column]).sum()
            match_percentage = (count / len(data)) * 100
            match_percentages.append(match_percentage)
        return pd.DataFrame({'RideTrack': columns, 'Accuracy (Total)': match_percentages})

    @staticmethod
    def _compute_label_accuracy(data, columns):
        class_labels = data['Action'].unique()
        match_percentages = []
        for column in columns:
            column_match_percentages = []
            for label in class_labels:
                total = (data['Action'] == label).sum()
                count = ((data['Action'] == label) & (data[column] == label)).sum()
                match_percentage = (count / total) * 100 if total != 0 else 0
                column_match_percentages.append(match_percentage)
            match_percentages.append(column_match_percentages)
        return pd.DataFrame(match_percentages, columns=class_labels, index=columns).reset_index()


    def calculate_action_prediction_counts(self, test_label, test_predict, draw_plot=False):
        conf_matrix = confusion_matrix(test_label, test_predict)

        labels = np.unique(test_label)
        columns = [f'Predicted: {label}' for label in labels]

        result_df = pd.DataFrame(columns=['Action'] + columns + ['Accuracy'])

        for i, action in enumerate(labels):
            true_label_count = conf_matrix[i, i]
            total_counts = conf_matrix[i, :].sum()
            accuracy = true_label_count / total_counts * 100 if total_counts != 0 else 0
            result_df.loc[i] = [action, *conf_matrix[i], accuracy]

        overall_correct = np.diag(conf_matrix).sum()
        overall_total = conf_matrix.sum()
        overall_accuracy = (overall_correct / overall_total) * 100 if overall_total != 0 else 0

        print(result_df.to_markdown(index=False))
    
        if draw_plot:
            self._draw_accuracy_plot(result_df, overall_accuracy)       

        return result_df

    def _draw_accuracy_plot(self, result_df, overall_accuracy):
        actions = result_df['Action'].tolist()

        fig, ax1 = plt.subplots(figsize=(10,6))

        palette = sns.color_palette("husl", len(actions))

        sns.barplot(x='Action', y='Accuracy', data=result_df, ax=ax1, palette=palette)

        ax1.axhline(overall_accuracy, color='red', linestyle='--')
        ax1.text(len(actions)-0.5, overall_accuracy + 1, f'Accuracy (Total): {overall_accuracy:.2f}', color='black', ha='right', fontsize=16)

        ax1.set_title('Accuracy analysis of different behaviors', fontsize=20, pad=12)
        ax1.set_xlabel('Behavior', fontsize=20, labelpad=10)
        ax1.set_ylabel('Accuracy', fontsize=20, labelpad=10)

        ax1.set_ylim(0, 100)
        plt.xticks(fontsize=14)
        for label in ax1.get_xticklabels():
            label.set_rotation(0)

        plt.show()
