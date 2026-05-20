import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load
from typing import Tuple, Optional, Union, List
from filterpy.kalman import KalmanFilter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class SensorFusion:

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
              ╔═══╗             ╔═══╗
              ║╔═╗║             ║╔══╝
              ║╚══╦══╦═╗╔══╦══╦═╣╚══╦╗╔╦══╦╦══╦═╗
              ╚══╗║║═╣╔╗╣══╣╔╗║╔╣╔══╣║║║══╬╣╔╗║╔╗╗
              ║╚═╝║║═╣║║╠══║╚╝║║║║  ║╚╝╠══║║╚╝║║║║
              ╚═══╩══╩╝╚╩══╩══╩╝╚╝  ╚══╩══╩╩══╩╝╚╝

        歡迎使用 RideTrack SensorFusion 功能！
        
        這個Class包含以下功能：
        
        1. Axis_Process: 用於處理來自車載 IMU 的數據。
           用法：Axis_Process(data_path, save_path)
        
        2. ECU_Reverse: 用於處理來自車載 ECU 的數據。
           用法：ECU_Reverse(data_path, save_path)
        
        3. Data_Merge: 用於合併兩個 CSV 檔案到一個檔案。
           用法：Data_Merge(ecu_data_path, axis_data_path, save_path)
        
        4. calibrate_angles: 用於校正角度數據。
           用法：calibrate_angles(dataset, save_path)

        5. calibrate_imu: 用於校正IMU數據。
           用法：calibrate_imu(dataset, k, save_path)

        6. normalize_data: 用於正規化指定特性。
           用法：normalize_data(dataset, feature, method, save_path)

        7. apply_kalman_filter: 應用卡爾曼濾波器到一個資料集。
           用法：apply_kalman_filter(dataset, features, q_noise, r_noise, save_path)

        8. apply_pca: 應用PCA。
            用法：apply_pca(df, n_components, save_model)

        9. get_feature_weights: 使用PCA獲得特徵權重。
            用法：get_feature_weights(df, pca_path)

        10. feature_importance: 使用隨機森林、XGBoost選擇重要特徵。
            用法：def feature_importance(self, X, y, encoder=None):
        
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


    # 處理IMU資料
    def Axis_Process(self, data_path: str, save_path: str) -> None:
        """
        Function: Used for processing data from a car-mounted Axis.

        Parameters:
            Data_Path: Path of the TXT file containing the data from the car-mounted device.
            Data_Save_Path: Path of the CSV file to save the processed data.

        Python Libraries:
            pandas: Used for handling CSV data.
            numpy: Used for performing scientific computing.
            tqdm: Used for displaying progress bars.
        """
        
        start_time = time.time()  # Start time

        Axis_Raw_Data = pd.read_csv(data_path, header=None)
        Reverse_Axis_Data_Feature = ["Absolute Time", "X-axis Angular Velocity", "Y-axis Angular Velocity", "Z-axis Angular Velocity", "X-axis Acceleration", "Y-axis Acceleration", "Z-axis Acceleration", "X-axis Angle", "Y-axis Angle", "Z-axis Angle"]
        Axis_Raw_Data = np.array(Axis_Raw_Data)
        row_lengh, column_lengh = Axis_Raw_Data.shape
        Axis_Raw_Data = Axis_Raw_Data.reshape(int(column_lengh/len(Reverse_Axis_Data_Feature)),len(Reverse_Axis_Data_Feature))
        
        
        Reverse_Axis_Data = []
        for x in range(int(column_lengh/len(Reverse_Axis_Data_Feature))):
            Reverse_Axis_Data.append(x)
        Reverse_Axis_Data = pd.DataFrame(columns = Reverse_Axis_Data_Feature ,index=Reverse_Axis_Data)
        
        
        print("\nReading 3-axis data in part1 (1/2)")
        for row in tqdm(range(int(column_lengh/len(Reverse_Axis_Data_Feature)))):
            for column in range(len(Reverse_Axis_Data_Feature)):    
                Reverse_Axis_Data.iloc[row][column] = Axis_Raw_Data[row][column]
        
        
        print("\nReading sampling time in part2 (2/2)")
        for row in tqdm(range (len(Reverse_Axis_Data)-1)):
            Reverse_Axis_Data['Absolute Time'][row] = Reverse_Axis_Data['Absolute Time'][row][2:len(Reverse_Axis_Data['Absolute Time'][row])]
            Reverse_Axis_Data['Absolute Time'].iloc[row] = pd.to_datetime(Reverse_Axis_Data['Absolute Time'].iloc[row],unit='ms',utc=True).tz_convert('Asia/Taipei') 
            Reverse_Axis_Data['Z-axis Angle'][row] = Reverse_Axis_Data['Z-axis Angle'][row][1:len(Reverse_Axis_Data['Z-axis Angle'][row])-1]
        
        Reverse_Axis_Data['Z-axis Angle'][(len(Reverse_Axis_Data)-1)] = Reverse_Axis_Data['Z-axis Angle'][(len(Reverse_Axis_Data)-1)][1:len(Reverse_Axis_Data['Z-axis Angle'][(len(Reverse_Axis_Data)-1)])-2]

        if save_path:
            try:
                self._save_dataframe(Reverse_Axis_Data, save_path)   
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")      

        self._print_execution_time(start_time)
        
        return Reverse_Axis_Data

    # 處理ECU資料，這段解碼有簽保密條款，公開時可以把這個Function拿掉
    def ECU_Reverse(self, data_path: str, save_path: Optional[str] = None) -> None:
        """
        Function: Used for processing data from a car-mounted ECU.

        Parameters:
            Data_Path: Path of the TXT file containing the data from the car-mounted device.
            Data_Save_Path: Path of the CSV file to save the processed data.

        Python Libraries:
            pandas: Used for handling CSV data.
            numpy: Used for performing scientific computing.
            tqdm: Used for displaying progress bars.
        """

        start_time = time.time()  # Start time

        ECU_Raw_Data = pd.read_csv(data_path, header=None)
        ECU_Raw_Data = ECU_Raw_Data.drop(ECU_Raw_Data.index[0:2])
        ECU_Raw_Data_0F = ECU_Raw_Data[ECU_Raw_Data.index%2 == 0 ]
        ECU_Raw_Data_0E = ECU_Raw_Data[ECU_Raw_Data.index%2 == 1 ]

        Reverse_ECU_Data_Feature = ["ECU Absolute Time", "Atmospheric Pressure", "Inclination Switch", "Fault Code Count", "Ignition Coil Current Diagnosis", "Fault Light Mileage", "Engine Operating Time", "Ignition Advance Angle", "Idling Correction Ignition Angle", "Fuel Injection Prohibition Mode", "Injection Mode", "Bypass Delay Correction", "ABV Opening", "ABV Idling Correction", "ABV Learning Value",  "Lambda Setting", "Air-Fuel Ratio Rich", "Closed Loop Control", "Air Flow", "Throttle Valve Air Flow", "Intake Manifold Pressure", "Intake Manifold Front Pressure", "MFF_AD_ADD_MMV_REL", "MFF_AD_FAC_MMV_REL", "MFF_AD_ADD_MMV", "MFF_AD_FAC_MMV", "Fuel Injection Quantity", "MFF_WUP_COR", "Ignition Mode", "Engine RPM", "Engine RPM Limit", "Idling Target RPM", "Fuel Injection Start Angle", "Fuel Pump State", "Engine State", "Engine Temperature", "Water Temperature PWM", "Ignition Magnetization Time", "Fuel Injection Time", "Closed Loop Fuel Correction","Intake Temperature", "Combustion Chamber Intake Temperature", "TPS Opening", "TPS Idling Learning Value", "Battery Voltage", "O2 Voltage", "Vehicle Speed", "TPS Voltage", "Seat Switch State"]
        Reverse_ECU_Data = []

        for row in range(min(len(ECU_Raw_Data_0E),len(ECU_Raw_Data_0F))):
            Reverse_ECU_Data.append(row)
            
        Reverse_ECU_Data = pd.DataFrame(columns = Reverse_ECU_Data_Feature ,index=Reverse_ECU_Data)


        print("\n【Reverse Engineering Restores ECU Data Part 1 (1/2)】")
        for row in tqdm(range(min(len(ECU_Raw_Data_0E),len(ECU_Raw_Data_0F)))):
            Reverse_ECU_Data['ECU Absolute Time'].iloc[row] = pd.to_datetime(ECU_Raw_Data_0E[2].iloc[row], unit='s',utc=True).tz_convert('Asia/Taipei')
            Reverse_ECU_Data['Intake Temperature'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][22:24]
            Reverse_ECU_Data['Combustion Chamber Intake Temperature'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][24:26]
            Reverse_ECU_Data['TPS Opening'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][26:30]
            Reverse_ECU_Data['TPS Idling Learning Value'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][30:34]
            Reverse_ECU_Data['Battery Voltage'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][34:36]
            Reverse_ECU_Data['O2 Voltage'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][36:40]
            Reverse_ECU_Data['Vehicle Speed'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][40:42]
            Reverse_ECU_Data['TPS Voltage'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][42:46]
            Reverse_ECU_Data['Seat Switch State'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][46:48]
            Reverse_ECU_Data['Inclination Switch'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][26:30]
            Reverse_ECU_Data['Fault Code Count'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][30:32]
            Reverse_ECU_Data['Ignition Coil Current Diagnosis'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][32:36]
            Reverse_ECU_Data['Fault Light Mileage'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][36:40]
            Reverse_ECU_Data['Engine Operating Time'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][40:44]
            Reverse_ECU_Data['Ignition Advance Angle'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][44:46]
            Reverse_ECU_Data['Idling Correction Ignition Angle'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][46:48]
            Reverse_ECU_Data['Fuel Injection Prohibition Mode'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][48:50]
            Reverse_ECU_Data['Injection Mode'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][50:52]   
            Reverse_ECU_Data['Bypass Delay Correction'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][52:54]
            Reverse_ECU_Data['ABV Opening'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][54:58]
            Reverse_ECU_Data['ABV Idling Correction'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][58:60]
            Reverse_ECU_Data['ABV Learning Value'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][60:62]
            Reverse_ECU_Data['Lambda Setting'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][62:64]
            Reverse_ECU_Data['Air-Fuel Ratio Rich'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][64:66]
            Reverse_ECU_Data['Closed Loop Control'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][66:68]
            Reverse_ECU_Data['Air Flow'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][68:72]
            Reverse_ECU_Data['Throttle Valve Air Flow'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][72:76]
            Reverse_ECU_Data['Intake Manifold Pressure'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][76:80]
            Reverse_ECU_Data['Intake Manifold Front Pressure'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][80:84]   
            Reverse_ECU_Data['MFF_AD_ADD_MMV_REL'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][84:88]
            Reverse_ECU_Data['MFF_AD_FAC_MMV_REL'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][88:92]
            Reverse_ECU_Data['MFF_AD_ADD_MMV'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][92:96]
            Reverse_ECU_Data['MFF_AD_FAC_MMV'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][96:100]    
            Reverse_ECU_Data['Fuel Injection Quantity'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][100:104]
            Reverse_ECU_Data['MFF_WUP_COR'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][104:106]
            Reverse_ECU_Data['Ignition Mode'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][106:108]
            Reverse_ECU_Data['Engine RPM'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][108:112]
            Reverse_ECU_Data['Engine RPM Limit'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][112:116]
            Reverse_ECU_Data['Idling Target RPM'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][116:120]
            Reverse_ECU_Data['Fuel Injection Start Angle'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][120:124]
            Reverse_ECU_Data['Fuel Pump State'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][124:126]
            Reverse_ECU_Data['Engine State'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][126:128]
            Reverse_ECU_Data['Engine Temperature'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][128:130]
            Reverse_ECU_Data['Water Temperature PWM'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][130:132]
            Reverse_ECU_Data['Ignition Magnetization Time'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][132:136]
            Reverse_ECU_Data['Fuel Injection Time'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][136:140]
            Reverse_ECU_Data['Closed Loop Fuel Correction'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][140:144]
            Reverse_ECU_Data['Atmospheric Pressure'].iloc[row] = ECU_Raw_Data_0E[1].iloc[row][22:26]
    
        print("\n【Reverse Engineering Restores ECU Data Part 2 (2/2)】")

        for row in tqdm(range(min(len(ECU_Raw_Data_0E),len(ECU_Raw_Data_0F)))): 
            Reverse_ECU_Data['ECU Absolute Time'].iloc[row] = Reverse_ECU_Data['ECU Absolute Time'].iloc[row]
            Reverse_ECU_Data['Atmospheric Pressure'].iloc[row] = int(Reverse_ECU_Data['Atmospheric Pressure'].iloc[row],16)
            Reverse_ECU_Data['Inclination Switch'].iloc[row] = int(Reverse_ECU_Data['Inclination Switch'].iloc[row],16)*0.004887107
            Reverse_ECU_Data['Fault Code Count'].iloc[row] = int(Reverse_ECU_Data['Fault Code Count'].iloc[row],16)
            Reverse_ECU_Data['Ignition Coil Current Diagnosis'].iloc[row] = int(Reverse_ECU_Data['Ignition Coil Current Diagnosis'].iloc[row],16)*0.004882796      
            Reverse_ECU_Data['Fault Light Mileage'].iloc[row] = int(Reverse_ECU_Data['Fault Light Mileage'].iloc[row],16)
            Reverse_ECU_Data['Engine Operating Time'].iloc[row] = int(Reverse_ECU_Data['Engine Operating Time'].iloc[row],16)*0.083333333  
            Reverse_ECU_Data['Ignition Advance Angle'].iloc[row] = (int(Reverse_ECU_Data['Ignition Advance Angle'].iloc[row],16)*0.468745098)-30
            Reverse_ECU_Data['Idling Correction Ignition Angle'].iloc[row] =  (int(Reverse_ECU_Data['Idling Correction Ignition Angle'].iloc[row],16)*0.468745098)-30 
            Reverse_ECU_Data['Fuel Injection Prohibition Mode'].iloc[row] =  int(Reverse_ECU_Data['Fuel Injection Prohibition Mode'].iloc[row],16)
            Reverse_ECU_Data['Injection Mode'].iloc[row] =  int(Reverse_ECU_Data['Injection Mode'].iloc[row],16)
            Reverse_ECU_Data['Bypass Delay Correction'].iloc[row] =  (int(Reverse_ECU_Data['Bypass Delay Correction'].iloc[row],16)*0.1)-12.8
            Reverse_ECU_Data['ABV Opening'].iloc[row] =  (int(Reverse_ECU_Data['ABV Opening'].iloc[row],16)*0.46875)
            Reverse_ECU_Data['ABV Idling Correction'].iloc[row] =  (int(Reverse_ECU_Data['ABV Idling Correction'].iloc[row],16)*0.937490196)-120
            Reverse_ECU_Data['ABV Learning Value'].iloc[row] =  (int(Reverse_ECU_Data['ABV Learning Value'].iloc[row],16)*0.937490196)-120
            Reverse_ECU_Data['Lambda Setting'].iloc[row] =  (int(Reverse_ECU_Data['Lambda Setting'].iloc[row],16)*0.003905882)+0.5
            Reverse_ECU_Data['Air-Fuel Ratio Rich'].iloc[row] =  int(Reverse_ECU_Data['Air-Fuel Ratio Rich'].iloc[row],16)                                                                   
            Reverse_ECU_Data['Closed Loop Control'].iloc[row] =  int(Reverse_ECU_Data['Closed Loop Control'].iloc[row],16)                                      
            Reverse_ECU_Data['Air Flow'].iloc[row] =  (int(Reverse_ECU_Data['Air Flow'].iloc[row],16)*0.015624994)    
            Reverse_ECU_Data['Throttle Valve Air Flow'].iloc[row] =  (int(Reverse_ECU_Data['Throttle Valve Air Flow'].iloc[row],16)*0.015624994)    
            Reverse_ECU_Data['Intake Manifold Pressure'].iloc[row] =  int(Reverse_ECU_Data['Intake Manifold Pressure'].iloc[row],16)
            Reverse_ECU_Data['Intake Manifold Front Pressure'].iloc[row] =  int(Reverse_ECU_Data['Intake Manifold Front Pressure'].iloc[row],16)                                                                          
            Reverse_ECU_Data['MFF_AD_ADD_MMV_REL'].iloc[row] =  (int(Reverse_ECU_Data['MFF_AD_ADD_MMV_REL'].iloc[row],16)*0.003906249)-128                                         
            Reverse_ECU_Data['MFF_AD_FAC_MMV_REL'].iloc[row] =  (int(Reverse_ECU_Data['MFF_AD_FAC_MMV_REL'].iloc[row],16)*0.000976562)-32
            Reverse_ECU_Data['MFF_AD_ADD_MMV'].iloc[row] =  (int(Reverse_ECU_Data['MFF_AD_ADD_MMV'].iloc[row],16)*0.003906249)-128                                         
            Reverse_ECU_Data['MFF_AD_FAC_MMV'].iloc[row] =  (int(Reverse_ECU_Data['MFF_AD_FAC_MMV'].iloc[row],16)*0.000976562)-32                                                                                 
            Reverse_ECU_Data['Fuel Injection Quantity'].iloc[row] =  (int(Reverse_ECU_Data['Fuel Injection Quantity'].iloc[row],16)*0.003906249)                                                                                                                           
            Reverse_ECU_Data['MFF_WUP_COR'].iloc[row] =  (int(Reverse_ECU_Data['MFF_WUP_COR'].iloc[row],16)*0.003905882)                                    
            Reverse_ECU_Data['Ignition Mode'].iloc[row] =  int(Reverse_ECU_Data['Ignition Mode'].iloc[row],16)  
            Reverse_ECU_Data['Engine RPM'].iloc[row] =  int(Reverse_ECU_Data['Engine RPM'].iloc[row],16)
            Reverse_ECU_Data['Engine RPM Limit'].iloc[row] =  int(Reverse_ECU_Data['Engine RPM Limit'].iloc[row],16)                                                                    
            Reverse_ECU_Data['Idling Target RPM'].iloc[row] =  int(Reverse_ECU_Data['Idling Target RPM'].iloc[row],16)-32768
            Reverse_ECU_Data['Fuel Injection Start Angle'].iloc[row] =  (int(Reverse_ECU_Data['Fuel Injection Start Angle'].iloc[row],16)*0.46875)-180                                                                           
            Reverse_ECU_Data['Fuel Pump State'].iloc[row] =  int(Reverse_ECU_Data['Fuel Pump State'].iloc[row],16)
            Reverse_ECU_Data['Engine State'].iloc[row] =  int(Reverse_ECU_Data['Engine State'].iloc[row],16)                                                                            
            Reverse_ECU_Data['Engine Temperature'].iloc[row] =  int(Reverse_ECU_Data['Engine Temperature'].iloc[row],16)-40                                                                                                                                             
            Reverse_ECU_Data['Water Temperature PWM'].iloc[row] =  (int(Reverse_ECU_Data['Water Temperature PWM'].iloc[row],16)*0.390588235)                                                                                
            Reverse_ECU_Data['Ignition Magnetization Time'].iloc[row] =  (int(Reverse_ECU_Data['Ignition Magnetization Time'].iloc[row],16)*0.004)                                                                                    
            Reverse_ECU_Data['Fuel Injection Time'].iloc[row] =  (int(Reverse_ECU_Data['Fuel Injection Time'].iloc[row],16)*0.004)                                                                                                               
            Reverse_ECU_Data['Closed Loop Fuel Correction'].iloc[row] =  (int(Reverse_ECU_Data['Closed Loop Fuel Correction'].iloc[row],16)*0.000976428)-32                                                                                                                                  
            Reverse_ECU_Data['Intake Temperature'].iloc[row] =  int(Reverse_ECU_Data['Intake Temperature'].iloc[row],16)-40                                        
            Reverse_ECU_Data['Combustion Chamber Intake Temperature'].iloc[row] = int(Reverse_ECU_Data['Combustion Chamber Intake Temperature'].iloc[row],16)-40                                                                                                                                                                                                     
            Reverse_ECU_Data['TPS Opening'].iloc[row] = (int(Reverse_ECU_Data['TPS Opening'].iloc[row],16)*0.001953124)                                        
            Reverse_ECU_Data['TPS Idling Learning Value'].iloc[row] = (int(Reverse_ECU_Data['TPS Idling Learning Value'].iloc[row],16)*0.004882796)                                                                                
            Reverse_ECU_Data['Battery Voltage'].iloc[row] = (int(Reverse_ECU_Data['Battery Voltage'].iloc[row],16)*0.062498039)+4                                                                                                                       
            Reverse_ECU_Data['O2 Voltage'].iloc[row] = (int(Reverse_ECU_Data['O2 Voltage'].iloc[row],16)*0.004882796)                                      
            #Reverse_ECU_Data['Vehicle Speed'].iloc[row] = (int(Reverse_ECU_Data['Vehicle Speed'].iloc[row],16)*0.594417404)  
            Reverse_ECU_Data['Vehicle Speed'].iloc[row] = (Reverse_ECU_Data['Engine RPM'].iloc[row]*60*434*3.14)/10000000
            Reverse_ECU_Data['TPS Voltage'].iloc[row] = (int(Reverse_ECU_Data['TPS Voltage'].iloc[row],16)*0.004882796)                                       
            Reverse_ECU_Data['Seat Switch State'].iloc[row] = int(Reverse_ECU_Data['Seat Switch State'].iloc[row],16)                                       
        
        if save_path:
            try:
                self._save_dataframe(Reverse_ECU_Data, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")            
        self._print_execution_time(start_time)

        return Reverse_ECU_Data

    # 合併資料
    def Data_Merge(self, ecu_data_path: str, axis_data_path: str, save_path: Optional[str] = None) -> None:
        """

        Function: used to merge two CSV files into one file.

        Parameters:
            
            ECU_Data_Path: the file path of the CSV file containing ECU data.

            Axis_Data_Path: the file path of the CSV file containing instrument data.

            Merge_Data_Path: the file path where the merged file will be stored.

        Libraries:

            pandas: used for CSV data processing.

            numpy: used for scientific computing.

            tqdm: used for displaying progress bar.

        """

        start_time = time.time()  # Start time

        ECU_Raw_Data = pd.read_csv(ecu_data_path)
        #ECU_Raw_Data = ECU_Raw_Data.drop('Unnamed: 0',axis=1)

        Axis_Raw_Data = pd.read_csv(axis_data_path)
        #Axis_Raw_Data = Axis_Raw_Data.drop('Unnamed: 0',axis=1)

    
        Merge_Data_No_Feature = ['No']
        Merge_Data_No = []
        for row in range(len(ECU_Raw_Data['ECU Absolute Time'])):
            Merge_Data_No.append(row)

        Merge_Data_No = pd.DataFrame(columns = Merge_Data_No_Feature ,index=Merge_Data_No)


        print ("\n【Data Engineering Megre Data Part 1 (1/2)】")
 
        for row in tqdm(range (len(ECU_Raw_Data['ECU Absolute Time'])-1)):
            Merge_Data_No['No'].iloc[row] = (Axis_Raw_Data['Absolute Time'] < ECU_Raw_Data['ECU Absolute Time'][row]).sum()

        Merge_Data_No = Merge_Data_No.fillna(0)


        Merge_Data_Number_Feature = ['Number']
        Merge_Data_Number = []
        for row in range(len(ECU_Raw_Data['ECU Absolute Time'])):
            Merge_Data_Number.append(row)

        Merge_Data_Number = pd.DataFrame(columns = Merge_Data_Number_Feature ,index=Merge_Data_Number)


        print ("\n【Data Engineering Megre Data Part 2 (2/2)】")

        for row in tqdm(range (len(ECU_Raw_Data['ECU Absolute Time'])-1)):
            Merge_Data_Number['Number'].iloc[row] = (Axis_Raw_Data['Absolute Time'] < ECU_Raw_Data['ECU Absolute Time'][row+1]).sum() - (Axis_Raw_Data['Absolute Time'] < ECU_Raw_Data['ECU Absolute Time'][row]).sum()

        Merge_Data_Number = Merge_Data_Number.fillna(0)

        Merge_ECU_Data_Feature = ["ECU Absolute Time", "Atmospheric Pressure", "Inclination Switch", "Fault Code Count", "Ignition Coil Current Diagnosis", "Fault Light Mileage", "Engine Operating Time", "Ignition Advance Angle", "Idling Correction Ignition Angle", "Fuel Injection Prohibition Mode", "Injection Mode", "Bypass Delay Correction", "ABV Opening", "ABV Idling Correction", "ABV Learning Value",  "Lambda Setting", "Air-Fuel Ratio Rich", "Closed Loop Control", "Air Flow", "Throttle Valve Air Flow", "Intake Manifold Pressure", "Intake Manifold Front Pressure", "MFF_AD_ADD_MMV_REL", "MFF_AD_FAC_MMV_REL", "MFF_AD_ADD_MMV", "MFF_AD_FAC_MMV", "Fuel Injection Quantity", "MFF_WUP_COR", "Ignition Mode", "Engine RPM", "Engine RPM Limit", "Idling Target RPM", "Fuel Injection Start Angle", "Fuel Pump State", "Engine State", "Engine Temperature", "Water Temperature PWM", "Ignition Magnetization Time", "Fuel Injection Time", "Closed Loop Fuel Correction", "Intake Temperature", "Combustion Chamber Intake Temperature", "TPS Opening", "TPS Idling Learning Value", "Battery Voltage", "O2 Voltage", "Vehicle Speed", "TPS Voltage", "Seat Switch State"]

        Merge_ECU_Data = []

    
        lenght = len(Axis_Raw_Data) - Merge_Data_No["No"].iloc[0]

        for row in range(lenght):
            Merge_ECU_Data.append(row)

        Merge_ECU_Data = pd.DataFrame(columns = Merge_ECU_Data_Feature ,index=Merge_ECU_Data)

        count=0
        for row in range(len(ECU_Raw_Data)):
            for column in range (int(Merge_Data_Number["Number"].iloc[row])):
                Merge_ECU_Data.iloc[count] = ECU_Raw_Data.iloc[row]
                count = count + 1 
        
        Merge_ECU_Data  = Merge_ECU_Data.reset_index(drop=True)


        Merge_ECU_Data = Merge_ECU_Data.dropna(axis=0)

        last = len(ECU_Raw_Data)-1
        Max = len(Axis_Raw_Data) -  (ECU_Raw_Data['ECU Absolute Time'][last] < Axis_Raw_Data['Absolute Time']).sum()

        Merge_Axis_Data = Axis_Raw_Data.iloc[Merge_Data_No['No'].iloc[0]:Max]
        Merge_Axis_Data = Merge_Axis_Data.reset_index(drop=True)
        Merge_Data = pd.concat([Merge_ECU_Data, Merge_Axis_Data], axis=1)

        if save_path:
            try:
                self._save_dataframe(Merge_Data, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")
        self._print_execution_time(start_time)

        return Merge_Data


    # 校正角度使用
    def calibrate_angles(self, dataset: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Function: Used for calibrating angle data.

        Parameters:
            dataset: DataFrame containing the angle data.
            save_path: Path of the CSV file to save the calibrated data.

        Python Libraries:
            pandas: Used for handling CSV data.
            numpy: Used for performing scientific computing.
        """
    
        start_time = time.time()  # Start time

        # Copy the dataset to prevent modifying the original one
        calibrated_data = dataset.copy()

        # Convert DataFrame to numpy array for efficiency
        angles_array = dataset[['X-axis Angle', 'Y-axis Angle', 'Z-axis Angle']].to_numpy()
        calibrated_angles_array = angles_array.copy()

        # Define the initial angles
        initial_angles = np.radians(angles_array[0, :])  # Convert to radians

        # Define the rotation matrix
        rotation_matrix = self.get_rotation_matrix(initial_angles)
        inv_rotation_matrix = np.linalg.inv(rotation_matrix)

        # Apply the inverse rotation matrix to each set of angles
        for i in tqdm(range(len(angles_array))):
            # Convert angles to radians
            angles = np.radians(angles_array[i, :])

            # Apply the inverse rotation matrix
            new_angles = np.dot(inv_rotation_matrix, angles)

            # Convert back to degrees and update the calibrated data
            calibrated_angles_array[i, :] = np.degrees(new_angles)

        # Update the DataFrame
        calibrated_data[['X-axis Angle', 'Y-axis Angle', 'Z-axis Angle']] = calibrated_angles_array

        if save_path:
            try:
                self._save_dataframe(calibrated_data, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")

        self._print_execution_time(start_time)

        return calibrated_data
    

    
    # 校正角度呼叫副程式
    def get_rotation_matrix(self, angles):
        """
        Function: Compute the rotation matrix.

        Parameters:
            angles: A numpy array containing the x, y, z angles in radians.
        """

        rotation_x = np.array([[1, 0, 0],
                               [0, np.cos(angles[0]), -np.sin(angles[0])],
                               [0, np.sin(angles[0]), np.cos(angles[0])]])

        rotation_y = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                               [0, 1, 0],
                               [-np.sin(angles[1]), 0, np.cos(angles[1])]])

        rotation_z = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                               [np.sin(angles[2]), np.cos(angles[2]), 0],
                               [0, 0, 1]])

        rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))

        return rotation_matrix



    # 校正加速度角速度使用
    def calibrate_imu(self, dataset: pd.DataFrame, k: int, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Function: Used for calibrating IMU data.

        Parameters:
            dataset: DataFrame containing the IMU data.
            k: Number of initial samples to use for calibration.
            save_path: Path of the CSV file to save the calibrated data.

        Python Libraries:
            pandas: Used for handling CSV data.
            numpy: Used for performing scientific computing.
        """
    
        start_time = time.time()  # Start time

        features = ['X-axis Angular Velocity', 'Y-axis Angular Velocity', 'Z-axis Angular Velocity', 
                    'X-axis Acceleration', 'Y-axis Acceleration', 'Z-axis Acceleration']
    
        # Copy the dataset to prevent modifying the original one
        calibrated_data = dataset.copy()  
    
        for feature in features:
            # Compute the mean of the first k samples
            mean_value = dataset[feature][:k].mean()
        
            # Subtract the mean from the entire column
            calibrated_data[feature] -= mean_value

        if save_path:
            try:
                self._save_dataframe(calibrated_data, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")

        self._print_execution_time(start_time)

        return calibrated_data


    def normalize_data(self, dataset: pd.DataFrame, feature: Union[str, List[str]], method: str = "minmax", save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Function: Normalize specified feature in dataset.

        Parameters:
            dataset: The dataframe containing the data to normalize.
            feature: The column(s) in the dataframe to normalize.
            method: The normalization method to use. Options are "minmax", "standard", "robust".
            save_path: Path to save normalized data. If None, data will not be saved.

        Returns:
            normalized_df: The dataframe after normalization.
        """
        start_time = time.time()  # Start time

        # 定義一個字典來映射方法名稱到相應的類
        methods = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "robust": RobustScaler
        }

        # 檢查指定的方法是否存在
        if method not in methods:
            raise ValueError(f"Invalid method. Expected one of: {list(methods.keys())}")

        # 創建相應的物件
        scaler = methods[method]()

        # 對指定特徵進行正規化
        normalized_data = scaler.fit_transform(dataset[feature])

        # 將正規化後的資料轉換為DataFrame
        normalized_df = pd.DataFrame(normalized_data, columns=feature)

        # 將DataFrame保存為CSV檔案
        if save_path:
            try:
                self._save_dataframe(normalized_df, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")

        self._print_execution_time(start_time)

        return normalized_df


    def initialize_kalman_filter(self, dim, q_noise=0.0001, r_noise=0.001):
        """Initializes a Kalman filter."""
    
        kf = KalmanFilter(dim_x=dim, dim_z=dim)
        kf.F = np.eye(dim)
        kf.H = np.eye(dim)
        kf.Q = np.eye(dim) * q_noise
        kf.R = np.eye(dim) * r_noise
        kf.x = np.zeros((dim, 1))
        kf.P = np.eye(dim)

        return kf

    def apply_kalman_filter(self, dataset: pd.DataFrame, features: Union[str, List[str]], q_noise: float = 0.0001, r_noise: float = 0.001, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Apply Kalman filter to a dataset.

        Parameters:
            dataset: DataFrame containing the data.
            features: Features to apply the filter on.
            q_noise: Noise in the system.
            r_noise: Measurement noise.
            save_path: Path of the CSV file to save the filtered data.
        """
    
        start_time = time.time()

        # Initialize the Kalman filter
        kf = self.initialize_kalman_filter(len(features), q_noise, r_noise)

        # Convert DataFrame to numpy array for efficiency
        data_array = dataset[features].to_numpy()
        filtered_data_array = np.zeros_like(data_array)

        # Apply the Kalman filter
        for i in tqdm(range(data_array.shape[0])):
            measurement = data_array[i, :].reshape(-1, 1)

            # Predict the next state
            kf.predict()

            # Update the state
            kf.update(measurement)

            # Save the filtered result
            filtered_data_array[i, :] = kf.x[:, 0]

        # Convert the filtered data back to DataFrame and update the original dataset
        filtered_df = pd.DataFrame(filtered_data_array, columns=features)
        for feature in features:
            dataset[feature] = filtered_df[feature]

        if save_path:
            try:
                self._save_dataframe(dataset, save_path)
            except Exception as e:
                print(f"Failed to save data to {save_path}: {e}")
        self._print_execution_time(start_time)

        return dataset

    def apply_pca(self, df: pd.DataFrame, n_components: Optional[int] = None, save_model: Optional[str] = None) -> Tuple[pd.DataFrame, PCA]:
        
        """
        Function: Apply PCA on a dataframe and optionally save the model.
        
        Parameters: 
            df: DataFrame. The dataset to apply PCA.
            n_components: int or None. The number of components to keep. 
                          If None, keep components that explain 95% of the variance.
            model_path: str. The path to save the PCA model.
        
        Returns: 
            df_pca: DataFrame. The transformed dataset.
            pca: PCA object. The PCA model used for transformation.
        """
        start_time = time.time()  # Start time
        # Determine the number of components
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(df)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= 0.95) + 1
        print(f'適合降至{n_components}維度')
        # Apply PCA
        pca = PCA(n_components=n_components)
        df_pca = pca.fit_transform(df)

        # Save the PCA model
        if save_model:
            dump(pca, save_model)

        self._print_execution_time(start_time)
        return df_pca, pca

    def get_feature_weights(self, df: pd.DataFrame, pca_path: str) -> pd.DataFrame:
        """
        Function: Calculate and print the weight of each feature based on the PCA model.

        Parameters: 
            df: DataFrame. The original dataset.
            pca_path: str. The path of the PCA model used for transformation.

        Returns: 
            feature_weights_df: DataFrame. Sorted weights of the features.
        """

        start_time = time.time()  # Start time
        # Load the PCA model
        pca = load(pca_path)

        # Multiply the components by the explained variance ratio
        weighted_components = pca.components_.T * pca.explained_variance_ratio_

        # Get the absolute sum of weights for each original feature
        feature_weights = np.sum(np.abs(weighted_components), axis=1)

        # Create a DataFrame for better visualization
        feature_weights_df = pd.DataFrame({
            'Feature': df.columns,
            'Weight': feature_weights
        })

        # Sort by weight
        feature_weights_df = feature_weights_df.sort_values(by='Weight', ascending=False)
        
        self._print_execution_time(start_time)
        return feature_weights_df

    def feature_importance(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], 
                           encoder: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:

        if encoder is None:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        else:
            y = encoder.transform(y)
        
        mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)
        importance_rf = pd.DataFrame({
            'Feature_Name': X.columns,
            'RF_Importance': rf.feature_importances_
        }).sort_values(by='RF_Importance', ascending=False)

        xgb = XGBClassifier()
        xgb.fit(X, y)
        importance_xgb = pd.DataFrame({
            'Feature_Name': X.columns,
            'XGB_Importance': xgb.feature_importances_
        }).sort_values(by='XGB_Importance', ascending=False)

        pca = PCA(n_components=X.shape[1])
        pca.fit(X)
        weighted_components = pca.components_.T * pca.explained_variance_ratio_
        feature_weights = np.sum(np.abs(weighted_components), axis=1)
        importance_pca = pd.DataFrame({
            'Feature_Name': X.columns,
            'PCA_Importance': feature_weights / np.sum(feature_weights)  # 正規化以使總和為1
        }).sort_values(by='PCA_Importance', ascending=False)

        return importance_rf, importance_xgb, importance_pca, mapping
