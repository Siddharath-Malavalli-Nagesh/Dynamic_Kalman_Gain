# Dynamic Kalman Gain using Kalman Net

## Steps to process data and train the model
Here are the steps  to run data fetch from nclt dataset. 
cd into 'Downloads' folder , 
and run -
```
python3 downloader.py --date "2012-01-08" --sen --gt --hokuyo
```

once complete run the nclt_preprocess.py file to preprocess data so that it can be fed directly into the Kalman Net model.
update line 115 if needed based on where you want to store .npz files 
```
    train_np = np.load('../Downloads/sensor_data/nclt_train.npz')
    val_np   = np.load('../Downloads/sensor_data/nclt_val.npz')
    test_np  = np.load('../Downloads/sensor_data/nclt_test.npz')
```
