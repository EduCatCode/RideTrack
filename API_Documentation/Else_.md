### [回目錄](../ReadMe.md)
# `else_` API

API提供一系列的數據處理方法，包括傳統的閥值分類和編碼以及繪製圖表功能。

## 1.class else_.Tradition_Category(輸入訓練集, 輸入分位數, 輸入特徵, 是否儲存處理後的檔案)
```python=
    def Tradition_Category(self, DataSet: pd.DataFrame, Quantiles: List[float], Feature: List[str], Save: bool = False) -> pd.DataFrame:
        """
        傳統閥值分類(Equal Width Bucketing)：對指定的特徵根據指定的分位數進行分類。
        
        Parameters
        ----------
            DataSet (pd.DataFrame):
                原始資料集。
            Quantiles (List[float]):
                要用於分類的分位數列表。
            Feature (List[str]):
                需要進行分類的特徵名稱列表。
            Save (bool, 可選):
                是否儲存分類後的資料集。預設為 False。
        
        Returns
        -------
            pd.DataFrame: 已經進行分類的資料集。

        Examples
        --------
        >>> else_ = RideTrack.else_()
        >>> original_data = pd.DataFrame({'Age': [20, 25, 30, 35, 40], 'Income': [30000, 40000, 50000, 60000, 70000]})
        >>> quantiles = [0.2, 0.4, 0.6, 0.8, 1]
        >>> features = ['Age', 'Income']
        >>> categorized_data = else_.Tradition_Category(original_data, quantiles, features)
        """
```

## 2.class else_.Tradition_Category_Value(輸入訓練集, 輸入分位數數值, 輸入特徵, 是否儲存處理後的檔案)

```python=
    def Tradition_Category_Value(self, DataSet: pd.DataFrame, Quantiles_Value: List[List[float]], Feature: List[str], Save: bool = False) -> pd.DataFrame:
        """
        傳統閥值分類(Equal Width Bucketing)：根據指定的數值對指定的特徵進行分類。
        
        Parameters
        ----------
            DataSet (pd.DataFrame):
                原始資料集。
            Quantiles_Value (List[List[float]]):
                要用於分類的數值列表。
            Feature (List[str]):
                需要進行分類的特徵名稱列表。
            Save (bool, 可選):
                是否儲存分類後的資料集。預設為 False。
        
        Returns
        -------
            pd.DataFrame: 已經進行分類的資料集。

        Examples
        --------
        >>> else_ = RideTrack.else_()
        >>> original_data = pd.DataFrame({'Age': [25, 45, 35, 50], 'Income': [50000, 100000, 70000, 150000]})
        >>> quantiles_value = [[20, 30, 40, 60], [30000, 60000, 100000, 200000]]
        >>> features = ['Age', 'Income']
        >>> categorized_data = else_.Tradition_Category_Value(original_data, quantiles_value, features, Save=False)
        
        """
```

## 3.class else_.Tradition_Encoding(輸入訓練集, 輸入特徵, 是否儲存)

```python=
    def Tradition_Encoding(self, DataSet: pd.DataFrame, Feature: List[str], Save: bool = False) -> pd.DataFrame:
        """
        傳統閥值編碼：根據指定特徵的值創建一個 'Action Element'。

        Parameters
        ----------
        DataSet (pd.DataFrame):
            原始資料集。
        Feature (List[str]):
            需要進行編碼的特徵名稱列表。
        Save (bool, optional):
            是否儲存編碼後的資料集。預設為 False。

        Returns
        -------
        pd.DataFrame:
            已經進行編碼的資料集。

        Examples
        --------
        >>> else_ = RideTrack.else_()
        >>> original_data = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [4, 5, 6]})
        >>> encoded_data = else_.Tradition_Encoding(original_data, ['Feature1', 'Feature2'])
        """
```

## 4.class else_.Tradition_Find_Top_K(輸入訓練集, 輸入目標百分比)

```python=
    def Tradition_Find_Top_K(self, DataSet: pd.DataFrame, Target_Percentage: float) -> pd.DataFrame:
        """
        找到佔資料集百分比最高的 Top K 類別。

        Parameters
        ----------
        DataSet : pd.DataFrame
            原始資料集，必須包含名為 'Action Element' 的列。
        Target_Percentage : float
            目標百分比。

        Returns
        -------
        pd.DataFrame
            包含類別統計信息的 DataFrame。

        Examples
        --------
        >>> else_ = RideTrack.else_()
        >>> original_data = pd.DataFrame({'Action Element': [1, 2, 3, 3, 2]})
        >>> stats = else_.Tradition_Find_Top_K(original_data, 80.0)
        """
```
## 5.class else_.Plot_Action_Cluster(輸入訓練集, 行為1, 行為2, 要觀察的特徵, 分群數, 是否儲存)

```python=
    def Plot_Action_Cluster(self, DataSet: pd.DataFrame, Action1: Union[int, str], Action2: Union[int, str], Feature: str, Cluster: int, Length: int, Save: bool = False) -> None:
        """
        繪製底層動作所組成之高階行為。

        Parameters
        ----------
        DataSet : pd.DataFrame
            原始資料集，必須包含名為 'Action' 和 'Action Element' 的列。
        Action1 : int or str
            第一種動作的標籤。
        Action2 : int or str
            第二種動作的標籤。
        Feature : str
            要繪製的特徵名。
        Cluster : int
            群集（或動作元素）的數量。
        Length : int
            每種動作要繪製的數據點數量。
        Save : bool, optional
            是否保存圖片。

        Examples
        --------
        >>> else_ = RideTrack.else_()
        >>> original_data = pd.DataFrame({'Action': ['Run', 'Walk', 'Run'], 'Action Element': [1, 2, 1], 'Speed': [10, 5, 12]})
        >>> else_.Plot_Action_Cluster(original_data, 'Run', 'Walk', 'Speed', 3, 100)
        """
```

## 6.class else_.Plot_Action_Track(輸入訓練集, 觀察特徵, 一張圖顯示多少個時間步, 是否儲存)
```python=
    def Plot_Action_Track(self, DataSet: pd.DataFrame, Step_Column_Name: str, Slice_size: int, Save: bool = False) -> None:
        """
        繪製駕駛行為軌跡。

        Parameters:
        -----------
        DataSet: pd.DataFrame
            包含駕駛行為資料的資料集。
        Step_Column_Name: str
            代表不同駕駛特徵名。
        Slice_size: int
            用於繪圖的每個資料切片的大小。
        Save: bool
            是否將繪圖儲存為圖像檔案。

        Returns:
        --------
        None

        Examples
        --------
        >>> else_ = RideTrack.else_()
        >>> else_.Plot_Action_Track(test_df, 'Filter_Predict', 500, False)
        """
```
## 7.class else_.Calculating_Time('影片軸ECU開始時間', 影片中標記點時間, '真實採樣時ECU開始時間')
```python=
    def Calculating_Time(self, Video_Ecu_Time: str, Video_Mark_Time: str, Real_Ecu_Time: str) -> str:
        """
        影片中標記換算時間使用。

        Parameters:
        -----------
        Video_Ecu_Time: str
            影片的 ECU 時間 (HH:MM:SS:FF)。
        Video_Mark_Time: str
            影片格式中的標記時間 (HH:MM:SS:FF)。
        Real_Ecu_Time: str
            實際 ECU 時間，格式為 (HH:MM:SS)。

        Returns:
        --------
        str
            計算出的真實世界標記時間。
        
        Examples
        --------
        >>> else_ = RideTrack.else_()
        >>> result = else_.Calculating_Time('00:00:10:01', '00:01:20:21', '12:30:45')
        """
```