# 環境建立
- 方法1: 使用 Anaconda yml檔案建置
```bash =
conda env create -f RideTrack_environment.yml
```

- 方法2: 使用 PyPI requirements建置
```bash=
conda create -n RideTrack python=3.7
pip install -r requirements.txt
```

- 方法3: 使用 Docker
```bash=
docker build -t RideTrack .
docker run RideTrack
```


# RideTrack 基本資訊

## 系統的主要功能與目的
RideTrack專注於主動式機車駕駛行為追蹤。該方法基於IMU（Inertial Measurement Unit）和ECU（Engine Control Unit）的感測數據，並不使用影像資料。這樣不僅能預測和辨識機車駕駛行為，還能在保護駕駛隱私的前提下實現。

## 開發團隊
本系統由 **NCHU KDD 721** 團隊負責開發。

## API 介紹
RideTrack系統主要包含以下模組：
* [SensorFusion: 用於感測器數據的融合與預處理。](./API_Documentation/SensorFusion.md)
* [AutoTag: 底層動作的自動標記。](./API_Documentation/AutoTag.md)
* [DrivePSTs: 用於處理駕駛上層行為的模型。](./API_Documentation/DrivePSTs.md)
* [Else: 傳統的閥值分類和編碼以及繪製圖表功能。](./API_Documentation/Else.md)
