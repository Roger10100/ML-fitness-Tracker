import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc=pd.read_csv("/home/ojas-srivastava/Desktop/data-science-template-main/data/raw/MetaMotion/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files= glob("/home/ojas-srivastava/Desktop/data-science-template-main/data/raw/MetaMotion/MetaMotion/")
len(files)


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path="/home/ojas-srivastava/Desktop/data-science-template-main/data/raw/MetaMotion/MetaMotion/*.csv"
f=files[0]
f.split("-")[0].replace(data_path,"")

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
def data_from_files(files):
    
    acc_df=pd.DataFrame()
    gyr_df=pd.DataFrame()
    acc_set=1
    gyr_set=1
    for f in files:
        participant=f.split("-")[4].split("/")[5].replace(data_path,"")
        category=f.split("-")[6].split("_")[0].rstrip("123")
        label=f.split("-")[5]
        # print(category)
        if "Accelerometer" in f:
            df["set"]=acc_set
            acc_set+=1
            acc_df= pd.concat([acc_df,df])
        if "Gyroscope" in f:
            df["set"]=gyr_set
            gyr_set+=1
            gyr_df= pd.concat([gyr_df,df])
    acc_df.index=pd.to_datetime(acc_df["epoch (ms)"],unit="ms")
    gyr_df.index=pd.to_datetime(gyr_df["epoch (ms)"],unit="ms")
    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]
    return acc_df,gyr_df
acc_df,gyr_df= data_from_files(files)    
acc_df



# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
