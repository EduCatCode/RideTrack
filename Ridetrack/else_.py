import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from filterpy.kalman import KalmanFilter

class else_:

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
                          ╭━━━┳╮
                          ┃╭━━┫┃
                          ┃╰━━┫┃╭━━┳━━╮
                          ┃╭━━┫┃┃━━┫┃━┫
                          ┃╰━━┫╰╋━━┃┃━┫
                          ╰━━━┻━┻━━┻━━╯

        歡迎使用 RideTrack Else 功能！
        
        這個Class包含以下功能：
        
        1. Tradition_Category: Equal Width Bucketing (輸入分位數)。
           用法：Tradition_Category(self, DataSet, Quantiles, Feature, Save)
        
        2. Tradition_Category_Value: Equal Width Bucketing (輸入數值)。
           用法：Tradition_Category_Value(self, DataSet, Quantiles_Value, Feature, Save)
        
        3. Tradition_Encoding: 依分類標記SnapShot。
           用法：Tradition_Encoding(self, DataSet, Feature, Save)

        4. 用法：Tradition_Find_Top_K: 找 Top K 佔資料百分比。
           用法：Tradition_Find_Top_K(self, DataSet, Target_Percentage)

        5. 用法：Plot_Action_Cluter: 繪製底層動作所組成之高階行為。
           用法：Plot_Action_Cluter(self, DataSet, Action1, Action2, Feature, Cluster, Length, Save)

        6. 用法：Plot_Action_Track: 繪製駕駛行為軌跡。
           用法：Plot_Action_Track(self, DataSet, Step_Column_Name, Slice_size, Save)

        7. Calculating_Time: 影片中標記換算時間使用。
           用法：Calculating_Time(self, Video_Ecu_Time, Video_Mark_Time, Real_Ecu_Time)

        """
        print(intro)


    # 傳統閥值分類並儲存檔案
    def Tradition_Category(self, DataSet, Quantiles, Feature, Save):

        Category=[]
        for feature in Feature:
            Variable_Category = f"{feature}_Category"
            Category.append(Variable_Category)
            Variable_thresholds = DataSet[feature].quantile(Quantiles).tolist()

            DataSet[Variable_Category] = pd.cut(DataSet[feature], bins=Variable_thresholds, labels=False)
    
        DataSet[Category] = DataSet[Category].fillna(0)

        if Save:
            DataSet.to_csv(f'Traditional_Threshold_{len(Quantiles)-1}^{len(Feature)}_groups.csv', index=False)
        
        return DataSet
    
    # 傳統閥值分類並儲存檔案
    def Tradition_Category_Value(self, DataSet, Quantiles_Value, Feature, Save):

        Category=[]
        for index, feature in enumerate(Feature):
            Variable_Category = f"{feature}_Category"
            Category.append(Variable_Category)
            Variable_thresholds = Quantiles_Value[index]

            DataSet[Variable_Category] = pd.cut(DataSet[feature], bins=Variable_thresholds, labels=False)
    
        DataSet[Category] = DataSet[Category].fillna(0)

        if Save:
            DataSet.to_csv(f'Traditional_Threshold_{len(Quantiles_Value)-1}^{len(Feature)}_groups.csv', index=False)
        
        return DataSet

    # Tradition_Encoding 呼叫使用轉成十進制
    def convert_to_decimal(self, Number, Base):
        decimal = 0
        power = 0
        while Number > 0:
            digit = Number % 10
            decimal += digit * (Base ** power)
            Number //= 10
            power += 1
        return decimal

    # 傳統閥值編碼並儲存檔案
    def Tradition_Encoding(self, DataSet, Feature, Save):
        for num in range(len(DataSet)):
            combined_string = ''.join(DataSet[Feature].iloc[num].astype(int).astype(str))
            converted_list = [int(char) for char in combined_string]
            max_value = max(converted_list)

            base = int(max_value) + 1
            decimal_number = self.convert_to_decimal(int(combined_string), base)
            DataSet['Action Element'].iloc[num] = decimal_number


        # 獲取當前日期和時間
        now = datetime.now()
        date = now.strftime('%Y%m%d')  # 格式化日期為YYYYMMDD
        time_str = now.strftime('%H%M')  # 格式化時間為HHMM

        if Save:
            DataSet.to_csv(f'{date}_Tradition_Encoding_{time_str}.csv', index=False)

        # 計算各類別的數量
        category_counts = DataSet['Action Element'].value_counts()

        # 繪製直方圖
        plt.bar(category_counts.index, category_counts.values)

        # 設定標題和軸標籤
        plt.title('Category Counts')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.savefig(f'{date}_plot_Category_Counts_{time_str}.png', dpi=300, bbox_inches='tight')
        plt.show()  
        plt.close()

        return DataSet   
 
    # 輸入資料含蓋量，取出前K個類別
    def Tradition_Find_Top_K(self, DataSet, Target_Percentage):
        # 計算各類別的數量
        category_counts = DataSet['Action Element'].value_counts()

        # 繪製直方圖
        plt.bar(category_counts.index, category_counts.values)

        # 設定標題和軸標籤
        plt.title('Category Counts')
        plt.xlabel('Category')
        plt.ylabel('Count')

        # 顯示圖形
        plt.show()


        # 根據數量由高到低排序
        category_counts = category_counts.sort_values(ascending=False)

        # 計算數量百分比
        category_percentages = category_counts / len(DataSet) * 100

        # 計算累積百分比
        category_cumulative_percentages = category_percentages.cumsum()

        # 建立 DataFrame
        category_stats = pd.DataFrame({'Count': category_counts, 'Percentage': category_percentages, 'Cumulative Percentage': category_cumulative_percentages})

        # 找到累積百分比達到目標百分比的資料
        filtered_data = category_stats[category_stats['Cumulative Percentage'] <= Target_Percentage]

        print(f'Data with cumulative percentage up to {Target_Percentage}%:')
        print(f'{filtered_data}\nTop K：{len(filtered_data)}')

        return category_stats

    # 繪製動作元素所組成之動作
    def Plot_Action_Cluter(self, DataSet, Action1, Action2, Feature, Cluster, Length, Save):    
        colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'cyan', 4: 'yellow', 5: 'magenta', 6: 'black', 7: 'white', 8: 'orange', 9: 'purple', 10: 'brown'}

        DataSet_Action1 = DataSet[DataSet['Action'] == Action1][:Length]
        DataSet_Action2 = DataSet[DataSet['Action'] == Action2][:Length]
    
        # 設定 x 軸長度
        DataSet_Action1_Length  = np.arange(len(DataSet_Action1))
        DataSet_Action2_Length = np.arange(len(DataSet_Action2))


        for i in range(Cluster):
            plt.scatter([], [], c=colors[i], label=f"Action Element {i}")

        # 點
        plt.scatter(DataSet_Action1_Length,  DataSet_Action1[Feature], c=[colors[x] for x in DataSet_Action1 ['Action Element']], zorder=2)

        # 線 
        plt.plot(DataSet_Action1_Length, DataSet_Action1[Feature][DataSet_Action1['Action'] == Action1], c='LightBlue' , label=Action1, linewidth=10, zorder=1)    

        plt.scatter(DataSet_Action2_Length, DataSet_Action2[Feature], c=[colors[x] for x in DataSet_Action2['Action Element']], zorder=2)

        plt.plot(DataSet_Action2_Length, DataSet_Action2[Feature][DataSet_Action2['Action'] == Action2], c='LightGreen' , label=Action2, linewidth=10, zorder=1)

        plt.legend(loc='best',bbox_to_anchor=(1.55, 1))
        plt.title(f'{Feature}\n {Action1} vs {Action2}')

        plt.xlabel('Time Step')
        plt.ylabel('Value')
        

        if Save:
            plt.savefig(f'{Feature} {Action1} vs {Action2}（best）.png', bbox_inches='tight')

        plt.show()

    # 繪製駕駛行為軌跡
    def Plot_Action_Track(self, DataSet, Step_Column_Name, Slice_size, Save):

        start_time = time.time()

        DataSet = DataSet.fillna('Unlabeled')

        # 計算需要切割的次數
        num_slices = math.ceil(len(DataSet) / Slice_size)

        # 動態計算切割範圍
        slices = []
        for i in range(num_slices):
            start = i * Slice_size
            end = min((i + 1) * Slice_size, len(DataSet))
            slices.append((start, end))

        # 設定顏色映射
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'cyan'] 

        # 繪製散點圖
        for i, (start, end) in enumerate(slices):
            # 切割資料
            Test_Data_slice = DataSet.iloc[start:end]
            Test_Data_slice.reset_index(drop=True, inplace=True)

            # 繪製第一張圖 (Test_Data_slice[Step_Column_Name])
            plt.figure(figsize=(12, 8))
            for j, condition in enumerate(['Go Straight', 'Idle', 'Turn Right', 'Turn Left', 'Two-Stage Left', 'U-turn', 'Unlabeled']):
                condition_points = [idx for idx, val in enumerate(Test_Data_slice[Step_Column_Name]) if val == condition]
                plt.scatter(Test_Data_slice.index[condition_points], Test_Data_slice['Z-axis Angular Velocity'][condition_points],
                            color=colors[j], label=condition, alpha=0.5)

            plt.title(f'{i} Slice - {Step_Column_Name}')
            plt.xlabel('Time Step')
            plt.ylabel('Z-axis Angular Velocity')
            plt.xticks(range(0, len(Test_Data_slice)+1, 500))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            if Save:
                plt.savefig(f'plot_{i+1}_step.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

            # 繪製第二張圖 (Test_Data_slice['Action'])
            plt.figure(figsize=(12, 8))
            for j, condition in enumerate(['Go Straight', 'Idle', 'Turn Right', 'Turn Left', 'Two-Stage Left', 'U-turn', 'Unlabeled']):
                condition_points = [idx for idx, val in enumerate(Test_Data_slice['Action']) if val == condition]
                plt.scatter(Test_Data_slice.index[condition_points], Test_Data_slice['Z-axis Angular Velocity'][condition_points],
                            color=colors[j], label=condition, alpha=0.5)

            plt.title(f'{i} Slice - Action')
            plt.xlabel('Time Step')
            plt.ylabel('Z-axis Angular Velocity')
            plt.xticks(range(0, len(Test_Data_slice)+1, 500))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            if Save:            
                plt.savefig(f'plot_{i+1}_action.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        # 計算執行時間
        end_time = time.time()
        execution_time = end_time - start_time
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = int(execution_time % 60)
        print(f"繪製動作序列所花費時間：{hours}小時{minutes}分鐘{seconds}秒")

        return

    # 影片中標記換算時間使用
    def Calculating_Time(self, Video_Ecu_Time, Video_Mark_Time, Real_Ecu_Time):


        Video_Ecu_Time_split = Video_Ecu_Time.split(':', 4)
        Hours_1   = int(Video_Ecu_Time_split[0])
        Minutes_1 = int(Video_Ecu_Time_split[1])
        Seconds_1 = int(Video_Ecu_Time_split[2])
        Frames_1  = int(Video_Ecu_Time_split[3])

        Video_Mark_Time_split = Video_Mark_Time.split(':', 4)
        Hours_2   = int(Video_Mark_Time_split[0])
        Minutes_2 = int(Video_Mark_Time_split[1])
        Seconds_2 = int(Video_Mark_Time_split[2])
        Frames_2  = int(Video_Mark_Time_split[3])

        Diff_Hours   = Hours_2   - Hours_1
        Diff_Minutes = Minutes_2 - Minutes_1
        Diff_Seconds = Seconds_2 - Seconds_1

        if Frames_2 >= Frames_1:
            Diff_Frames = Frames_2 - Frames_1
        else:
            Diff_Seconds = Diff_Seconds - 1 
            Diff_Frames = Frames_2 - Frames_1 + 25
    

        Real_Ecu_Time_dt = datetime.strptime(Real_Ecu_Time, "%H:%M:%S")

        real_diff_time = timedelta(hours=Diff_Hours, minutes=Diff_Minutes, seconds=Diff_Seconds)
        real_mark_time_dt = Real_Ecu_Time_dt + real_diff_time

        real_mark_time_str = real_mark_time_dt.strftime('%H:%M:%S')


        real_mark_time_split = real_mark_time_str.split(':', 3)
        Hours   = int(real_mark_time_split[0])
        Minutes = int(real_mark_time_split[1])
        Seconds = float(real_mark_time_split[2])


        real_time_error = 0.68
        frames_time_error = Diff_Frames*0.04

        Seconds = Seconds + real_time_error + frames_time_error

        real_mark_time = str(Hours)+':'+str(Minutes)+':'+str(Seconds)
    
        return real_mark_time

    # 簡易平滑資料使用
    def Convolve(self, Data_Set, File_Name, Data, Window_Size, Save):
        """
        Function: This function performs smoothing on the input feature data by replacing the original data with the average value within a window of size Window_Size. It plots the data of turning left and right in different colors on the same graph, and saves the result as a file.
    
        Parameters:
            Data_Set: pandas DataFrame, contains the feature and label data of the dataset
            File_Name: string, used to name the saved image file
            Data: numpy array, the numerical values of feature data
            Window_Size: integer, the size of the smoothing window
        """        
        Smoothed_Data = np.convolve(Data, np.ones(Window_Size)/Window_Size, mode='same')
        
        x1 = np.arange(len(Smoothed_Data[Data_Set['Action']== 'left']))
        x2 = np.arange(len(Smoothed_Data[Data_Set['Action']== 'right']))

        plt.figure()
        plt.plot(x1, Smoothed_Data[Data_Set['Action']== 'left'] , c='r' , label='Turn left')
        plt.plot(x2, Smoothed_Data[Data_Set['Action'] == 'right'], c='g', label='Turn right')
        plt.legend(loc='lower right')
        
        if Save:
            plt.savefig(File_Name+'_Window_Size_'+str(Window_Size)+'.png')
        
        return 

    def Data_Smoothing(self, DataSet, Feature, Method, Slice_Size):
        if Method == 'Kalman':
            kf = KalmanFilter(dim_x=len(Feature), dim_z=len(Feature))
            kf.F = np.eye(len(Feature))
            kf.H = np.eye(len(Feature))
            kf.Q = np.eye(len(Feature)) * 0.0001
            kf.R = np.eye(len(Feature)) * 0.001
            kf.x = np.zeros((len(Feature), 1))
            kf.P = np.eye(len(Feature))
            Data = np.array(DataSet[Feature].values)
            filtered_data = np.zeros_like(Data)

            for i in range(Data.shape[0]):
                measurement = Data[i, :].reshape(len(Feature), 1)
                kf.predict()
                kf.update(measurement)
                filtered_data[i, :] = kf.x[:, 0]

            for i, column in enumerate(Feature):
                DataSet[column] = filtered_data[:, i]


            num_slices = math.ceil(len(DataSet) / Slice_Size)
            slices = []
            for i in range(num_slices):
                start = i * Slice_Size
                end = min((i + 1) * Slice_Size, len(DataSet))
                slices.append((start, end))

            for i, (start, end) in enumerate(slices):
                plt.figure()
                Test_Data_slice = DataSet.iloc[start:end]
                Test_Data_slice.reset_index(drop=True, inplace=True)
                Feature_Value = Test_Data_slice[Feature]
                plt.title(f'{i} Slice - {Method}')
                plt.xlabel('Time Step')
                plt.ylabel(f'{Feature}')
                plt.plot(range(len(Test_Data_slice)), Feature_Value, label=f'{Feature}')

            return DataSet

        if Method == 'Window_Average':
            Window_Size = 30
            Smoothed_Data = np.convolve(DataSet[Feature], np.ones(Window_Size)/Window_Size, mode='same')
            DataSet[Feature] = Smoothed_Data


            num_slices = math.ceil(len(DataSet) / Slice_Size)
            slices = []
            for i in range(num_slices):
                start = i * Slice_Size
                end = min((i + 1) * Slice_Size, len(DataSet))
                slices.append((start, end))

            for i, (start, end) in enumerate(slices):
                plt.figure()
                Test_Data_slice = DataSet.iloc[start:end]
                Test_Data_slice.reset_index(drop=True, inplace=True)
                Feature_Value = Test_Data_slice[Feature]
                plt.title(f'{i} Slice - {Method}')
                plt.xlabel('Time Step')
                plt.ylabel(f'{Feature}')
                plt.plot(range(len(Test_Data_slice)), Feature_Value, label=f'{Feature}')

            return DataSet
        
        if Method == 'None':
            num_slices = math.ceil(len(DataSet) / Slice_Size)
            slices = []
            for i in range(num_slices):
                start = i * Slice_Size
                end = min((i + 1) * Slice_Size, len(DataSet))
                slices.append((start, end))

            for i, (start, end) in enumerate(slices):
                plt.figure()
                Test_Data_slice = DataSet.iloc[start:end]
                Test_Data_slice.reset_index(drop=True, inplace=True)
                Feature_Value = Test_Data_slice[Feature]
                plt.title(f'{i} Slice - {Method}')
                plt.xlabel('Time Step')
                plt.ylabel(f'{Feature}')
                plt.plot(range(len(Test_Data_slice)), Feature_Value, label=f'{Feature}')

            return DataSet
