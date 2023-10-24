### [回目錄](../ReadMe.md)

# `DrivePSTs` API

`DrivePSTs` API提供VoMM/PST用於處理駕駛上層行為的模型建立與預測。
參考網址: https://github.com/rpgomez/vomm

## 1.class DrivePSTs.train_vomm(輸入訓練集, 輸入PST的最大深度, 輸入底層分群數, 輸入儲存模型路徑)
```python=
    def train_vomm(self, train_data: pd.DataFrame, l: int, k: int, 
                   save_model: Optional[str] = None) -> None:
        """
        使用 VoMM（Variable Order Markov Model）或 PST（Probabilistic Suffix Tree）訓練模型，
        並根據指定的動作進行訓練。

        參數
        ----------
        train_data : pd.DataFrame
            包含訓練數據的數據框。
        l : int
            PST 的深度。
        k : int
            字母表大小。
        save_model : str, 可選
            保存訓練模型的路徑。如果為 None，模型將不會被保存。

        使用的庫
        --------------
        pandas : 用於處理表格數據。
        pickle : 用於保存訓練的模型。
        time : 用於計算執行時間。

        範例
        --------
        >>> DrivePSTs = RideTrack.DrivePSTs()
        >>> DrivePSTs.train_vomm(train_data, 3, 4, save_model="model.pkl")
        """
```

## 2.class DrivePSTs.test_vomm(輸入訓練集, 輸入取樣頻率, 輸入儲存模型路徑)
```python=
    def test_vomm(self, data_set: pd.DataFrame, frequency: int, 
                  save_path: Optional[str] = None) -> pd.DataFrame:
        """
        使用訓練過的 VoMM（Variable Order Markov Model）模型進行預測，
        並進行簡單的噪聲過濾。

        參數
        ----------
        data_set : pd.DataFrame
            包含待預測數據的數據框。
        frequency : int
            用於過濾噪聲的頻率閾值。
        save_path : str, 可選
            保存預測結果的數據集的路徑。如果為 None，數據將不會被保存。

        返回
        -------
        pd.DataFrame
            包含預測和過濾結果的數據框。

        使用的庫
        --------------
        pandas : 用於處理表格數據。
        time : 用於計算執行時間。
        math : 用於數學運算。
        tqdm : 用於進度條顯示。

        範例
        --------
        >>> DrivePSTs = RideTrack.DrivePSTs()
        >>> predicted_data = DrivePSTs.test_vomm(data_set, 10, save_path="predicted_data.csv")
        """
```
## 3.class DrivePSTs.compute_accuracy(輸入訓練集, 輸入取樣頻率, 輸入計算完檔案路徑, 是否要繪製出圖表)
```python=    
    def compute_accuracy(self, dataset: pd.DataFrame, frequency: int, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        計算各種駕駛行為的預測準確度。

        Parameters
        ----------
        dataset : pd.DataFrame
            包含實際行為和預測行為的數據集。
        frequency : int
            用於過濾行為的頻率。
        save_path : str, optional
            預測準確度的結果存儲路徑。


        Returns
        -------
        pd.DataFrame
            含有各種駕駛行為的預測準確度。

        Examples
        --------
        >>> DrivePSTs = RideTrack.DrivePSTs()
        >>> accuracy_df = DrivePSTs.compute_accuracy(dataset, 10, save_path="accuracy.csv")
        """
```



## 4.class DrivePSTs.calculate_action_prediction_counts(測試集label, 預測的Label)
```python=
    def calculate_action_prediction_counts(self, test_label: List[Union[str, int]], test_predict: List[Union[str, int]], draw_plot: bool = False) -> pd.DataFrame:
        """
        計算各種駕駛行為的預測次數和混淆矩陣。

        Parameters
        ----------
        test_label : List[Union[str, int]]
            測試數據的實際標籤。
        test_predict : List[Union[str, int]]
            測試數據的預測標籤。
        draw_plot : bool, optional
            是否繪製預測準確度的圖表。

        Returns
        -------
        pd.DataFrame
            含有各種駕駛行為的預測次數和準確度。

        Examples
        --------
        >>> DrivePSTs = RideTrack.DrivePSTs()
        >>> confusion_matrix_df = DrivePSTs.calculate_action_prediction_counts(test_label, test_predict, draw_plot=True)
        """
```