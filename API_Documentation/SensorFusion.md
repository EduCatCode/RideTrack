### [回目錄](../ReadMe.md)

# `SensorFusion` API

`SensorFusion` API提供一系列用於感測器數據的融合與預處理

## 1.class SensorFusion.Axis_Process(處理IMU的.txt檔案路徑, 處理完IMU的.csv儲存的檔案路徑)

```python=
def Axis_Process(self, data_path: str, save_path: str) -> None:
    """
    用於處理來自車載 IMU 的數據。
    
    Parameters
    ----------
    data_path : str
        IMU數據的文件路徑。
    save_path : str
        處理後的數據保存路徑。

    Returns
    -------
    None
        這個方法不返回任何值，但會將處理後的數據保存到`save_path`。

    Examples
    --------
    >>> fusion = RideTrack.SensorFusion()
    >>> fusion.Axis_Process('path_to_data.csv', 'path_to_save.csv')
    """
```


## 2.class SensorFusion.ECU_Reverse(處理ECU的.txt檔案路徑, 處理完ECU的.csv儲存的檔案路徑)
```python=
    def ECU_Reverse(self, data_path: str, save_path: Optional[str] = None) -> None:
        """
        用於處理來自車載 ECU 的數據。
        
        Parameters
        ----------
        data_path : str
            來自車載設備的TXT文件的路徑。
        save_path : Optional[str], default=None
            處理後數據保存的CSV文件的路徑。如果為None，則不保存數據。
        
        Returns
        -------
        None
            這個方法不返回任何值，但如果`save_path`被指定，會將處理後的數據保存到該路徑。
        
        Libraries Used
        ---------------
        pandas : 用於處理CSV數據。
        numpy  : 用於進行科學計算。
        tqdm   : 用於顯示進度條。

        Examples
        --------
        >>> fusion = RideTrack.SensorFusion()
        >>> fusion.ECU_Reverse('path_to_data.txt', 'path_to_save.csv')
        """
```




## 3.class SensorFusion.Data_Merge(處理完後ECU的檔案路徑, 處理完後IMU的檔案路徑, 合併完後.csv儲存的檔案路徑)
```python=
    def Data_Merge(self, ecu_data_path: str, axis_data_path: str, save_path: Optional[str] = None) -> None:
        """
        用於合併兩個 CSV 檔案到一個檔案。
        
        Parameters
        ----------
        ecu_data_path : str
            包含ECU數據的CSV文件的路徑。
        axis_data_path : str
            包含儀器數據的CSV文件的路徑。
        save_path : Optional[str], default=None
            合併後的文件將被存儲的文件路徑。如果為None，則不保存數據。
        
        Returns
        -------
        None
            這個方法不返回任何值，但如果`save_path`被指定，會將合併後的數據保存到該路徑。
        
        Libraries Used
        ---------------
        pandas : 用於CSV數據處理。
        numpy  : 用於進行科學計算。
        tqdm   : 用於顯示進度條。

        Examples
        --------
        >>> fusion = RideTrack.SensorFusion()
        >>> fusion.Data_Merge('ecu_data.csv', 'axis_data.csv', 'merged_data.csv')
        """
```



## 4.class SensorFusion.calibrate_angles('輸入要校正的資料集檔案路徑', '輸入校正完要儲存.csv檔案的路徑')

```python=
    def calibrate_angles(self, dataset: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        用於校正角度數據。
        
        Parameters
        ----------
        dataset : pd.DataFrame
            包含角度數據的DataFrame。
        save_path : Optional[str], default=None
            校正後數據將被保存的CSV文件路徑。如果為None，則不保存數據。
        
        Returns
        -------
        pd.DataFrame
            返回校正後的角度數據，以DataFrame的形式。
        
        Libraries Used
        ---------------
        pandas : 用於處理CSV數據。
        numpy  : 用於進行科學計算。

        Examples
        --------
        >>> fusion = RideTrack.SensorFusion()
        >>> df = pd.DataFrame({'angles': [0, 45, 90]})
        >>> calibrated_df = fusion.calibrate_angles(df, 'calibrated_angles.csv')
        """
  
```



## 5.class SensorFusion.calibrate_imu('輸入要校正的資料集檔案路徑', '輸入校正完要儲存.csv檔案的路徑') 
``` python=
    def calibrate_imu(self, dataset: pd.DataFrame, k: int, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        用於校正IMU數據。
        
        Parameters
        ----------
        dataset : pd.DataFrame
            包含IMU數據的DataFrame。
        k : int
            用於校正的初始樣本數量。
        save_path : Optional[str], default=None
            校正後數據將被保存的CSV文件路徑。如果為None，則不保存數據。
        
        Returns
        -------
        pd.DataFrame
            返回校正後的IMU數據，以DataFrame的形式。
        
        Libraries Used
        ---------------
        pandas : 用於處理CSV數據。
        numpy  : 用於進行科學計算。

        Examples
        --------
        >>> fusion = RideTrack.SensorFusion()
        >>> df = pd.DataFrame({'accel': [0.1, 0.2], 'gyro': [1.2, 1.3]})
        >>> calibrated_df = fusion.calibrate_imu(df, 5, 'calibrated_imu.csv')
        """

```



## 6.class SensorFusion.normalize_data('輸入要校正的資料集檔案路徑', '輸入要校正的特徵欄', '選擇要使用的正規化方法','輸入校正完要儲存.csv檔案的路徑') 

``` python=
    def normalize_data(self, dataset: pd.DataFrame, feature: Union[str, List[str]], method: str = "minmax", save_path: Optional[str] = None) -> pd.DataFrame:
        """
        用於正規化指定特性。
        
        Parameters
        ----------
        dataset : pd.DataFrame
            包含需要正規化的數據的DataFrame。
        feature : Union[str, List[str]]
            需要正規化的DataFrame中的列。
        method : str, default="minmax"
            要使用的正規化方法。選項有 "minmax"，"standard"，"robust"。
        save_path : Optional[str], default=None
            保存正規化數據的路徑。如果為None，數據將不會被保存。
        
        Returns
        -------
        pd.DataFrame
            正規化後的數據，以DataFrame的形式返回。
        
        Libraries Used
        ---------------
        pandas : 用於處理CSV數據。

        Examples
        --------
        >>> fusion = RideTrack.SensorFusion()
        >>> df = pd.DataFrame({'feature1': [0, 1, 2], 'feature2': [2, 1, 0]})
        >>> normalized_df = fusion.normalize_data(df, 'feature1', 'minmax', 'normalized_data.csv')
        """

```



## 7.class SensorFusion.apply_kalman_filter('輸入要校正的資料集檔案路徑', '輸入要校正的特徵欄', '輸入Q參數', '輸入R參數', '輸入校正完要儲存.csv檔案的路徑') 

``` python=
    def apply_kalman_filter(self, dataset: pd.DataFrame, features: Union[str, List[str]], q_noise: float = 0.0001, r_noise: float = 0.001, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        應用卡爾曼濾波器到一個資料集。
        
        Parameters
        ----------
        dataset : pd.DataFrame
            包含數據的DataFrame。
        features : Union[str, List[str]]
            需要應用濾波器的特性。
        q_noise : float, default=0.0001
            系統中的噪聲。
        r_noise : float, default=0.001
            測量噪聲。
        save_path : Optional[str], default=None
            保存濾波後數據的路徑。如果為None，數據將不會被保存。
        
        Returns
        -------
        pd.DataFrame
            濾波後的數據，以DataFrame的形式返回。
        
        Libraries Used
        ---------------
        pandas : 用於處理CSV數據。

        Examples
        --------
        >>> fusion = RideTrack.SensorFusion()
        >>> df = pd.DataFrame({'feature1': [0, 1, 2], 'feature2': [2, 1, 0]})
        >>> filtered_df = fusion.apply_kalman_filter(df, ['feature1', 'feature2'], 0.0001, 0.001, 'filtered_data.csv')
        """      
```





## 8.class SensorFusion.apply_pca('輸入要處理的資料集檔案路徑', '輸入要壓縮的維度(None:預設95%變異)', '輸入處理完要儲存.csv檔案的路徑') 

```python=
    def apply_pca(self, df: pd.DataFrame, n_components: Optional[int] = None, save_model: Optional[str] = None) -> Tuple[pd.DataFrame, PCA]:
        """
        應用PCA到一個數據集上，並且選項性保存模型。
        
        Parameters
        ----------
        df : pd.DataFrame
            要應用PCA的數據集。
        n_components : Optional[int], default=None
            保留的主成分數量。如果為None，則保留解釋95%方差的主成分。
        save_model : Optional[str], default=None
            保存PCA模型的路徑。
        
        Returns
        -------
        Tuple[pd.DataFrame, PCA]
            第一個元素是轉換後的數據集，以DataFrame的形式。
            第二個元素是用於轉換的PCA模型。
        
        Libraries Used
        ---------------
        pandas : 用於處理CSV數據。
        sklearn.decomposition : 用於PCA分解。

        Examples
        --------
        >>> fusion = RideTrack.SensorFusion()
        >>> df = pd.DataFrame({'feature1': [0, 1, 2], 'feature2': [2, 1, 0]})
        >>> transformed_df, pca_model = fusion.apply_pca(df, 2, 'pca_model.pkl')
        """

```

## 9.class SensorFusion.get_feature_weights('輸入要處理的資料集檔案路徑', '載入模型檔案的路徑') 

```python=
    def get_feature_weights(self, df: pd.DataFrame, pca_path: str) -> pd.DataFrame:
        """
        獲得特徵權重基於PCA模型。
        
        Parameters
        ----------
        df : pd.DataFrame
            原始的數據集。
        pca_path : str
            用於轉換的PCA模型的路徑。
        
        Returns
        -------
        pd.DataFrame
            特徵權重的DataFrame，經過排序。

        Libraries Used
        ---------------
        pandas : 用於處理CSV數據。

        Examples
        --------
        >>> fusion = RideTrack.SensorFusion()
        >>> df = pd.DataFrame({'feature1': [0, 1, 2], 'feature2': [2, 1, 0]})
        >>> feature_weights = fusion.get_feature_weights(df, 'pca_model.pkl')
        """

```


## 10.class SensorFusion.feature_importance('輸入要處理的資料集(特徵)', '輸入要處理的資料集(標籤)') 

```python=
    def feature_importance(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], encoder: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
        """
        使用隨機森林、PCA、XGBoost選擇重要特徵。
        
        Parameters
        ----------
        X : pd.DataFrame
            輸入特徵的數據集。
        y : pd.Series or pd.DataFrame
            目標變數。
        encoder : str, optional
            用於對目標變數進行編碼的方法，默認為None。

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            返回一個包含重要特徵的DataFrame以及一個包含模型評估信息的字典。
            
        Libraries Used
        ---------------
        pandas : 用於處理CSV數據。
        sklearn : 用於機器學習模型和數據處理。

        Examples
        --------
        >>> fusion = RideTrack.SensorFusion()
        >>> X = pd.DataFrame({'feature1': [0, 1, 2], 'feature2': [2, 1, 0]})
        >>> y = pd.Series([0, 1, 1])
        >>> important_features, model_info = fusion.feature_importance(X, y)
        """
```