### [回目錄](../ReadMe.md)

# `AutoTag` API

`AutoTag` API提供底層動作的自動標記與分群數建議

## 1.class AutoTag.cluster_data(輸入當前的資料.csv, 輸入特徵, 使用的分群演算法, 設定分群數, 儲存模型路徑, 分群完的.csv儲存的檔案路徑)


```python=
    def cluster_data(self, dataset: pd.DataFrame, feature: Union[str, List[str]], method: str = "kmeans", n_clusters: int = 3, model_path: str = "model.pkl", save_path: Optional[str] = None) -> pd.DataFrame:
        """
        在指定的特性（或特性集）上對數據集進行分群，並選擇性保存模型。
        
        參數
        ----------
        dataset : pd.DataFrame
            包含待分群數據的數據集。
        feature : str 或 List[str]
            需要進行分群的數據集中的列名或列名列表。
        method : str, 默認='kmeans'
            使用的分群算法。支持的算法有 "kmeans" 和 "agglomerative"。
        n_clusters : int, 默認=3
            要形成的分群數量。適用於 "kmeans" 和 "agglomerative"。
        model_path : str, 默認='model.pkl'
            保存訓練好的分群模型的路徑。
        save_path : str, 可選
            保存分群後數據集的路徑。如果為 None，數據集將不會被保存。
        
        返回
        -------
        pd.DataFrame
            包含新列，該列顯示每個樣本點的分群分配。
        
        使用的庫
        --------------
        pandas : 用於處理表格數據。
        sklearn : 用於分群算法。
        joblib : 用於保存訓練好的模型。

        範例
        --------
        >>> AutoTag = RideTrack.AutoTag()
        >>> clustered_data = AutoTag.cluster_data(dataset, '['特徵1', '特徵2'], n_clusters=4)
        """
```


## 2.class AutoTag.predict_cluster(輸入當前的資料.csv, 輸入特徵, 載入模型路徑, 預測分群完的.csv儲存的檔案路徑)
```python=
    def predict_cluster(self, dataset: pd.DataFrame, feature: Union[str, List[str]], model_path: str, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        使用預訓練模型對數據集中指定特性進行分群預測。
        
        參數
        ----------
        dataset : pd.DataFrame
            包含待預測數據的數據集。
        feature : str 或 List[str]
            需要進行預測的數據集中的列名或列名列表。
        model_path : str
            預訓練分群模型的路徑。
        save_path : str, 可選
            保存預測後數據集的路徑。如果為 None，數據集將不會被保存。
        
        返回
        -------
        pd.DataFrame
            預測後的數據集，包含一個新列顯示每個數據點的分群預測。
        
        使用的庫
        --------------
        pandas : 用於處理表格數據。

        範例
        --------
        >>> AutoTag = RideTrack.AutoTag()
        >>> predicted_data = AutoTag.predict_cluster(dataset, ['特徵1', '特徵2'], 'model_path.pkl')
        """
```

## 3.class AutoTag.determine_optimal_clusters(輸入當前的資料.csv, 輸入測試最大分群數, 分群效能結果.csv儲存的檔案路徑)

```python=
    def determine_optimal_clusters(self, dataset: pd.DataFrame, max_k: int, save_path: Optional[str] = None) -> Tuple[pd.DataFrame, dict, set]:
        """
        使用多種評估指標來確定最佳的分群數量。

        參數
        ----------
        dataset : pd.DataFrame
            包含待分群數據的數據集。
        max_k : int
            考慮的最大分群數。
        save_path : str, 可選
            保存評分數據的路徑。如果為 None，數據將不會被保存。

        返回
        -------
        pd.DataFrame
            包含各種評估指標和相應 k 值的數據框。
        dict
            各評估方法推薦的前半部分最佳 k 值。
        set
            根據前三個評估方法共同推薦的分群數。

        使用的庫
        --------------
        pandas : 用於處理表格數據。
        numpy : 用於數學運算。
        tqdm : 用於顯示進度條。
        time : 用於計算執行時間。

        範例
        --------
        >>> AutoTag = RideTrack.AutoTag()
        >>> scores, top_k, intersection = AutoTag.determine_optimal_clusters(dataset, 10)
        """
```