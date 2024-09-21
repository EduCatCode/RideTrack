# RideTrack - 機車駕駛行為追蹤系統

## 主要成就

![napkin-selection (7)](https://hackmd.io/_uploads/ry3wvPhaC.png)

在本專案中，我們開發了一套新穎且主動的機車駕駛行為追蹤系統RideTrack。此系統使用IMU（Inertial Measurement Unit）和ECU（Engine Control Unit）的感測數據來準確辨識駕駛行為，並有效保護駕駛隱私。以下是一些具體的量化成果：

- **隱私保護**：系統完全不依賴影像資料或GPS數據，實現高效隱私保護。
- **行為辨識準確率**：在實驗中，系統對各種駕駛行為的總體辨識準確率達到93%，顯著優於基於DeepConvLSTM架構的方案。即使在考慮行為之間的轉換銜接時間時，整體預測的準確度仍能維持在70%。
- **功能多樣性**：能夠辨識包括直行、左轉、右轉、迴轉、怠速、待轉等多種駕駛行為，比傳統IMU判斷方法更為全面。
- **即時預測能力**：系統能在駕駛過程中即時進行預測，預測延遲低於0.5秒。

## 目錄

1. [專案簡介](#專案簡介)
2. [專案目標](#專案目標)
3. [環境建立](#環境建立)
4. [系統模組](#系統模組)
5. [系統介紹](#系統介紹)
   - [預處理階段](#預處理階段)
   - [即時預測階段](#即時預測階段)
6. [實驗與結果](#實驗與結果)
   - [實驗數據的收集方法](#實驗數據的收集方法)
   - [資料集介紹](#資料集介紹)
   - [比較對象](#比較對象)
   - [性能比較](#性能比較)
7. [未來發展](#未來發展)

## 專案簡介

RideTrack專注於主動式機車駕駛行為追蹤。該系統基於IMU和ECU的感測數據，並不使用影像資料。這樣不僅能預測和辨識機車駕駛行為，還能在保護駕駛隱私的前提下實現。

## 專案目標

- **主要目標**: 通過分析IMU和ECU數據，實現對機車駕駛行為的準確辨識，並保護駕駛隱私。
- **次要目標**: 提升駕駛行為辨識的準確性和實時性，相較於傳統和深度學習方法有明顯優勢。

## 環境建立

### 方法1: 使用 Anaconda yml檔案建置

```bash =
conda env create -f RideTrack_environment.yml
```

### 方法2: 使用 PyPI requirements建置

```bash=
conda create -n RideTrack python=3.7
pip install -r requirements.txt
```

### 方法3: 使用 Docker

```bash=
docker build -t RideTrack .
docker run RideTrack
```



## 系統模組
RideTrack系統主要包含以下模組：
* [SensorFusion: 用於感測器數據的融合與預處理。](./API_Documentation/SensorFusion.md)
* [AutoTag: 底層動作的自動標記。](./API_Documentation/AutoTag.md)
* [DrivePSTs: 用於處理駕駛上層行為的模型。](./API_Documentation/DrivePSTs.md)
* [Else: 傳統的閥值分類和編碼以及繪製圖表功能。](./API_Documentation/Else.md)


## 開發團隊
本系統由 **NCHU KDD 721** 團隊負責開發。

## API 介紹
RideTrack系統主要包含以下模組：
* [SensorFusion: 用於感測器數據的融合與預處理。](./API_Documentation/SensorFusion.md)
* [AutoTag: 底層動作的自動標記。](./API_Documentation/AutoTag.md)
* [DrivePSTs: 用於處理駕駛上層行為的模型。](./API_Documentation/DrivePSTs.md)
* [Else: 傳統的閥值分類和編碼以及繪製圖表功能。](./API_Documentation/Else.md)


## 系統介紹
我們的團隊提出了一個新穎且主動的機車駕駛行為追蹤系統，名為「RideTrack」。該系統包含了兩個主要階段，並分為五個核心模塊。

![螢幕擷取畫面 2024-06-12 104439](https://hackmd.io/_uploads/SkqDUsUB0.png)

## 預處理階段
首先，在「SensorFusion」模塊中，我們進行了多項資料預處理步驟。透過這些步驟，我們能有效地提煉出資料特徵，進而提升了預測的準確性。接著，在「AutoTagIntegrator」模塊，我們對這些提煉出的資料特徵進行自動標記，將每一個快照（snapshot）標註成一個的底層機車狀態。在標記完成後，我們將這些已標記的資料轉換為序列標籤，然後在「DrivePSTs」模塊中針對不同的行為訓練不同的（PST）模型。我們會將訓練完成的所有模型儲存下來。

## 即時預測階段
進入即時預測階段，我們開始進行即時預測。在「LiveStreamer」模塊中，我們將接收到的資料進行預處理，並標記底層機車狀態，然後將這些資料存入「ActionPulsePredictor」的佇列中。利用已載入的PST狀態轉換模型進行預測，通過這種方法，我們在保護駕駛隱私的同時，實現了全面且即時地追蹤和監控機車駕駛的行為。

![245471200-7131ed94-7970-4043-8a56-4f904263e264](https://hackmd.io/_uploads/BJJ5djUr0.png)


## 實驗與結果

### 實驗數據的收集方法
- 駕駛穿戴GoPro攝影機記錄騎車的過程（作為後續標記參考）。
- 展示ECU接收器與IMU感測器的放置位置，如將Android平板放置在機車車箱中央（晃動最小的地方）以收集IMU資料。
- 實際採樣的畫面。

![螢幕擷取畫面 2024-06-12 104526](https://hackmd.io/_uploads/Sk2DUiLHA.png)


## 資料集介紹

資料集1來自學區附近，是在人流車流較少的下午2點所採集（如圖所示）。

這個資料集包含了：
採集10次直線、左轉、右轉、迴轉
20次怠速、待轉
因為在怠速（停紅燈）時時間較長，為了平衡各動作的快照數量，我們對怠速數據進行了下采樣（DownSample），在計算準確度時採用1/7的機率。

![圖片1](https://hackmd.io/_uploads/SkfhLjLS0.jpg)

資料集2來自夜市附近，是在人流車流較多的下午點6所採集（如圖所示）。

這個資料集包含了：
採集10次直線、4次的左轉、右轉
1次迴轉、6次怠速、1次待轉

![圖片2](https://hackmd.io/_uploads/SJQ38jUH0.jpg)

## 比較對象
我們的比較對象為DeepConvLSTM架構，結合了CNN和LSTM並加上Attention進行探討，也有Baseline的一些機器學習分類模型。我們的方法準確度與其他方法相比有明顯的優勢。

![螢幕擷取畫面 2024-06-12 104755](https://hackmd.io/_uploads/HJkd8s8HR.png)

## 性能比較

|準確度的性能比較|分類模型性能在snapshot和windows訓練模式下的比較|
|:-:|:-:|
|![圖片1](https://hackmd.io/_uploads/Bkp8Lj8rR.png)|![圖片2](https://hackmd.io/_uploads/B11DIiLSC.png)|


## 未來發展
未來我們將繼續優化RideTrack系統，並考慮以下發展方向：
- **數據集擴展**：我們計劃增加更多不同條件下的數據集，例如來自不同駕駛者、不同地理環境和天氣條件的數據，以提高模型的泛化能力，讓系統能在更廣泛的情境中精確運作。這不僅可以提升系統的準確性，還能讓模型適應更多變的真實世界駕駛行為。
- **模型改進**：未來我們計劃融合更先進的深度學習算法，同時保留目前演算法的優點。具體而言，我們將探索結合圖神經網絡（Graph Neural Networks）和強化學習（Reinforcement Learning）的方法，來進一步提升駕駛行為辨識的精確性和可靠性。這樣的改進將使得模型在處理複雜的駕駛行為時更加靈活和高效。
- **即時監控**：開發即時監控模組，實現在線預測和即時回饋。
