import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import dump, load
from typing import Tuple, Optional, Union, List
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class AutoTag:

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
                    ╭━━━╮  ╭╮  ╭━━━━╮
                    ┃╭━╮┃ ╭╯╰╮ ┃╭╮╭╮┃
                    ┃┃ ┃┣╮┣╮╭╋━┻┫┃┃┣┻━┳━━╮
                    ┃╰━╯┃┃┃┃┃┃╭╮┃┃┃┃╭╮┃╭╮┃
                    ┃╭━╮┃╰╯┃╰┫╰╯┃┃┃┃╭╮┃╰╯┃
                    ╰╯ ╰┻━━┻━┻━━╯╰╯╰╯╰┻━╮┃
                                      ╭━╯┃
                                      ╰━━╯

        歡迎使用 RideTrack AutoTag 功能！
        
        這個Class包含以下功能：
        
        1. cluster_data: 自動標記底層動作(分群)。
           用法：cluster_data(self, dataset, feature, method="kmeans", n_clusters=3, model_path="model.pkl", save_path=None)
        
        2. predict_cluster: 重置實驗使用。
           用法：predict_cluster(self, dataset, feature, model_path, save_path=None)
        
        3. determine_optimal_clusters: 尋找最佳分群數。
           用法：determine_optimal_clusters(self, dataset, max_k, save_path=None):
        
        
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


    #分群使用
    def cluster_data(self, dataset: pd.DataFrame, feature: Union[str, List[str]], method: str = "kmeans", n_clusters: int = 11, model_path: str = "model.pkl", save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Function: Perform clustering on specified feature in dataset.

        Parameters:
            dataset: The dataframe containing the data to cluster.
            feature: The column(s) in the dataframe to cluster.
            method: The clustering method to use. Options are "kmeans", "agglomerative", "dbscan".
            n_clusters: The number of clusters to form.
            model_path: Path to save clustering model.
            save_path: Path to save clustered data. If None, data will not be saved.

        Returns:
            dataset: The dataframe after clustering.
        """
        start_time = time.time()  # Start time

        # 定義一個字典來映射方法名稱到相應的類
        methods = {
            "kmeans": KMeans,
            "agglomerative": AgglomerativeClustering,
        }

        # 檢查指定的方法是否存在
        if method not in methods:
            raise ValueError(f"Invalid method. Expected one of: {list(methods.keys())}")

        # 創建相應的物件
        model = methods[method](n_clusters=n_clusters)

        # 訓練模型並進行分群
        model.fit(dataset[feature])

        # 分群結果
        dataset['Action Element'] = model.labels_

        # 儲存分群模型
        if model_path:
            dump(model, model_path)

        # 儲存分群完資料成 CSV 檔案
        if save_path:
            try:
                self._save_dataframe(dataset, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")
        self._print_execution_time(start_time)

        return dataset
    

    # 利用載入訓練好分群模型進行預測
    def predict_cluster(self, dataset: pd.DataFrame, feature: Union[str, List[str]], model_path: str, save_path: Optional[str] = None) -> pd.DataFrame:
        
        """
        Function: Predict cluster for specified feature in dataset using pre-trained model.

        Parameters:
            dataset: The dataframe containing the data to predict.
            feature: The column(s) in the dataframe to predict.
            model_path: Path to pre-trained clustering model.
            save_path: Path to save predicted data. If None, data will not be saved.

        Returns:
            dataset: The dataframe after prediction.
        """
        
        start_time = time.time()  # Start time

        # 載入模型
        model = load(model_path)

        # 預測
        dataset['Action Element'] = model.predict(dataset[feature])
        
        # 儲存預測後的資料成 CSV 檔案
        if save_path:
            try:
                self._save_dataframe(dataset, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")
        
        self._print_execution_time(start_time)

        return dataset


    # 計算分群最佳群數  副程式
    def evaluate_clustering(self, k, dataset):
        """
        Evaluate clustering performance.
        """
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(dataset)
        scores = {
            'Silhouette Score': silhouette_score(dataset, labels),
            'Calinski-Harabasz Index': calinski_harabasz_score(dataset, labels),
            'Davies-Bouldin Index': davies_bouldin_score(dataset, labels),
            'Distortion': kmeans.inertia_
        }
        return scores

    def determine_optimal_clusters(self, dataset: pd.DataFrame, max_k: int, save_path: Optional[str] = None) -> Tuple[pd.DataFrame, dict, set]:
        """
        Determine the optimal number of clusters using various evaluation metrics.
        """
        start_time = time.time()
        half_max_k = max_k // 2
        evaluation_methods = ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index', 'Distortion']
        
        # Initialize the scores list for each evaluation method
        scores = {method: [] for method in evaluation_methods}

        for k in tqdm(range(2, max_k+1)):
            new_scores = self.evaluate_clustering(k, dataset)
            for method in evaluation_methods:
                scores[method].append(new_scores[method])

        top_k = {}
        for method in evaluation_methods[:3]:  # Only the first 3 methods aim to be maximized
            top_k[method] = np.argsort(scores[method])[-half_max_k:] + 2  # +2 because k starts from 2
        top_k['Distortion'] = np.argsort(scores['Distortion'])[:half_max_k] + 2  # Distortion should be minimized

        # Calculate the intersection of the top half max_k clusters for the first 3 evaluation metrics
        intersection = set(top_k[evaluation_methods[0]])
        for method in evaluation_methods[1:3]:  # Exclude the 'Distortion' method
            intersection.intersection_update(top_k[method])

        df_scores = pd.DataFrame(scores, index=range(2, max_k+1))
        self._plot_scores(df_scores, save_path)  # Moved plotting to a separate method for clarity

        for method, ks in top_k.items():
            print(f"根據 {method}，前 {half_max_k} 個建議的分群數量分別為 {ks}")

        print(f"根據前三個評分標準推薦的分群數交集為 {intersection}")

        self._print_execution_time(start_time)
        return df_scores, top_k, intersection
    
    # 副程式
    def _plot_scores(self, df_scores, save_path):
        """
        Plot the scores for each clustering evaluation method.
        """
        plt.figure(figsize=(12, 10))
        for i, method in enumerate(df_scores.columns, 1):
            plt.subplot(2, 2, i)
            plt.plot(df_scores.index, df_scores[method], marker='o')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel(method)
            plt.title(method)
            plt.xticks(df_scores.index)

        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path)
            except Exception as e:
                print(f"Failed to save figure to {save_path}: {e}")
