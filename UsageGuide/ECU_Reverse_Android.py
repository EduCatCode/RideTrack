import time
import math
from typing import Tuple, Optional, Union, Dict, List
import numpy as np 
import pandas as pd
from collections import Counter
from tqdm import tqdm, trange
from datetime import datetime, timedelta



# 處理ECU資料，這段解碼有簽保密條款，公開時可以把這個Function拿掉
def ECU_Reverse(data_path: str, save_path: Optional[str] = None) -> None:
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


    ECU_Raw_Data = pd.read_csv(data_path, header=None)
    ECU_Raw_Data = ECU_Raw_Data.drop(ECU_Raw_Data.index[0:2])
    ECU_Raw_Data_0F = ECU_Raw_Data[ECU_Raw_Data.index%2 == 0 ]
    ECU_Raw_Data_0E = ECU_Raw_Data[ECU_Raw_Data.index%2 == 1 ]

    Reverse_ECU_Data_Feature = ["Atmospheric Pressure", "Inclination Switch", "Fault Code Count", "Ignition Coil Current Diagnosis", "Fault Light Mileage", "Engine Operating Time", "Ignition Advance Angle", "Idling Correction Ignition Angle", "Fuel Injection Prohibition Mode", "Injection Mode", "Bypass Delay Correction", "ABV Opening", "ABV Idling Correction", "ABV Learning Value",  "Lambda Setting", "Air-Fuel Ratio Rich", "Closed Loop Control", "Air Flow", "Throttle Valve Air Flow", "Intake Manifold Pressure", "Intake Manifold Front Pressure", "MFF_AD_ADD_MMV_REL", "MFF_AD_FAC_MMV_REL", "MFF_AD_ADD_MMV", "MFF_AD_FAC_MMV", "Fuel Injection Quantity", "MFF_WUP_COR", "Ignition Mode", "Engine RPM", "Engine RPM Limit", "Idling Target RPM", "Fuel Injection Start Angle", "Fuel Pump State", "Engine State", "Engine Temperature", "Water Temperature PWM", "Ignition Magnetization Time", "Fuel Injection Time", "Closed Loop Fuel Correction","Intake Temperature", "Combustion Chamber Intake Temperature", "TPS Opening", "TPS Idling Learning Value", "Battery Voltage", "O2 Voltage","Vehicle Speed" , "TPS Voltage", "Seat Switch State"]
    Reverse_ECU_Data = []

    for row in range(min(len(ECU_Raw_Data_0E),len(ECU_Raw_Data_0F))):
        Reverse_ECU_Data.append(row)
            
    Reverse_ECU_Data = pd.DataFrame(columns = Reverse_ECU_Data_Feature ,index=Reverse_ECU_Data)


    print("\n【Reverse Engineering Restores ECU Data Part 1 (1/2)】")
    for row in tqdm(range(min(len(ECU_Raw_Data_0E),len(ECU_Raw_Data_0F)))):
        #Reverse_ECU_Data['ECU Absolute Time'].iloc[row] = pd.to_datetime(ECU_Raw_Data_0E[2].iloc[row], unit='s',utc=True).tz_convert('Asia/Taipei')
        Reverse_ECU_Data['Intake Temperature'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][0:2]
        Reverse_ECU_Data['Combustion Chamber Intake Temperature'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][2:4]
        Reverse_ECU_Data['TPS Opening'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][4:8]
        Reverse_ECU_Data['TPS Idling Learning Value'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][8:12]
        Reverse_ECU_Data['Battery Voltage'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][12:14]
        Reverse_ECU_Data['O2 Voltage'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][14:18]
        Reverse_ECU_Data['Vehicle Speed'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][18:20]
        Reverse_ECU_Data['TPS Voltage'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][20:24]
        Reverse_ECU_Data['Seat Switch State'].iloc[row] = ECU_Raw_Data_0F[0].iloc[row][24:28]

        Reverse_ECU_Data['Atmospheric Pressure'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][4:8]
        Reverse_ECU_Data['Inclination Switch'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][8:12]
        Reverse_ECU_Data['Fault Code Count'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][12:14]
        Reverse_ECU_Data['Ignition Coil Current Diagnosis'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][14:18]
        Reverse_ECU_Data['Fault Light Mileage'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][18:22]
        Reverse_ECU_Data['Engine Operating Time'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][22:26]
        Reverse_ECU_Data['Ignition Advance Angle'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][26:28]
        Reverse_ECU_Data['Idling Correction Ignition Angle'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][28:30]
        Reverse_ECU_Data['Fuel Injection Prohibition Mode'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][30:32]
        Reverse_ECU_Data['Injection Mode'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][32:34]   
        Reverse_ECU_Data['Bypass Delay Correction'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][34:36]
        Reverse_ECU_Data['ABV Opening'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][36:40]
        Reverse_ECU_Data['ABV Idling Correction'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][40:44]
        Reverse_ECU_Data['ABV Learning Value'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][44:46]
        Reverse_ECU_Data['Lambda Setting'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][46:48]
        Reverse_ECU_Data['Air-Fuel Ratio Rich'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][48:50]
        Reverse_ECU_Data['Closed Loop Control'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][50:52]
        Reverse_ECU_Data['Air Flow'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][52:56]
        Reverse_ECU_Data['Throttle Valve Air Flow'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][56:60]
        Reverse_ECU_Data['Intake Manifold Pressure'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][60:64]
        Reverse_ECU_Data['Intake Manifold Front Pressure'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][64:68]   
        Reverse_ECU_Data['MFF_AD_ADD_MMV_REL'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][68:72]
        Reverse_ECU_Data['MFF_AD_FAC_MMV_REL'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][72:76]
        Reverse_ECU_Data['MFF_AD_ADD_MMV'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][76:80]
        Reverse_ECU_Data['MFF_AD_FAC_MMV'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][80:84]    
        Reverse_ECU_Data['Fuel Injection Quantity'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][84:88]
        Reverse_ECU_Data['MFF_WUP_COR'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][88:90]
        Reverse_ECU_Data['Ignition Mode'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][90:92]
        Reverse_ECU_Data['Engine RPM'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][92:96]
        Reverse_ECU_Data['Engine RPM Limit'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][96:100]
        Reverse_ECU_Data['Idling Target RPM'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][100:104]
        Reverse_ECU_Data['Fuel Injection Start Angle'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][104:108]
        Reverse_ECU_Data['Fuel Pump State'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][108:110]
        Reverse_ECU_Data['Engine State'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][110:112]
        Reverse_ECU_Data['Engine Temperature'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][112:114]
        Reverse_ECU_Data['Water Temperature PWM'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][114:116]
        Reverse_ECU_Data['Ignition Magnetization Time'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][116:120]
        Reverse_ECU_Data['Fuel Injection Time'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][120:124]
        Reverse_ECU_Data['Closed Loop Fuel Correction'].iloc[row] = ECU_Raw_Data_0E[0].iloc[row][124:128]


    
    print("\n【Reverse Engineering Restores ECU Data Part 2 (2/2)】")

    for row in tqdm(range(min(len(ECU_Raw_Data_0E),len(ECU_Raw_Data_0F)))): 
        #Reverse_ECU_Data['ECU Absolute Time'].iloc[row] = Reverse_ECU_Data['ECU Absolute Time'].iloc[row]
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
        Reverse_ECU_Data['Vehicle Speed'].iloc[row] = (int(Reverse_ECU_Data['Vehicle Speed'].iloc[row],16)*0.594417404)  
        #Reverse_ECU_Data['Vehicle Speed'].iloc[row] = (Reverse_ECU_Data['Engine RPM'].iloc[row]*60*434*3.14)/10000000
        Reverse_ECU_Data['TPS Voltage'].iloc[row] = (int(Reverse_ECU_Data['TPS Voltage'].iloc[row],16)*0.004882796)                                       
        Reverse_ECU_Data['Seat Switch State'].iloc[row] = int(Reverse_ECU_Data['Seat Switch State'].iloc[row],16)                                       
        
    if save_path:
        try:
            Reverse_ECU_Data.to_csv(save_path)
        except Exception as e:
            print(f"Failed to save data to {save_path}: {e}")            

    return 


if __name__ == "__main__":
    data_path = str(input('input data_path'))
    save_path = str(input('input save_path'))
    ECU_Reverse(data_path, save_path)