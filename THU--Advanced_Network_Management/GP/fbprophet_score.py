import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import f1_score
import glob
import os


# Updates df with new scores from model
def update_score(df,df_test,pred):
    df_test.index = (pd.to_datetime(df_test.index).astype(int)/ 10**9).astype(int)
    print(df_test.shape, len(pred))
    df_test['predict'] = pred
    
    print(df_test.groupby(df_test.predict).count())
    # temp = df_test.groupby(df_test.predict).count()

    # if temp.shape[0] > 1:
        # print(temp.iloc[1].value / temp.iloc[0].value)
    # print('HERE3')
    
    # Create Dictionary
    rename_dict = df_test.set_index(['KPI ID'],append=True).to_dict()['predict']

    # Map Values based on multi-index
    df = df['predict'].update(pd.Series(rename_dict))



def naive_classifier(df_test, up_limit=None, down_limit=None):
    # pred = [0 for i in range(df_test.shape[0])]
    pred = 0

    if up_limit != None and down_limit == None:
        pred = np.where( (df_test.value.values > up_limit), 1, 0 )
    elif up_limit == None and down_limit != None:
        pred = np.where( (df_test.value.values < down_limit), 1, 0 )
    else:
        pred = np.where( ((df_test.value.values < down_limit) | (df_test.value.values > up_limit) ), 1, 0 )
    
    return pred

def naive_classifier_extra(df_test, limit):
    arr = df_test.value.values
    pred = [0 for x in range(len(arr))]

    for i in range(len(arr)):
        if arr[i] > limit:
            pred[i-1] = 1
            pred[i] = 1
            pred[i+1] = 1

    return pred

def naive_adjuster(pred, df_test, limiter, up=True):
    arr = df_test['value'].values
    if up:
        for i in range(len(pred)):
            if arr[i]  > limiter:
                pred[i] = 1
    else:
        for i in range(len(pred)):
            if arr[i] < limiter:
                pred[i] = 1

    return pred

def naive_test(pred,fraction,threshold):
    for i in range(len(fraction)):
        if fraction[i] < threshold:
            pred[i] = 1

    return pred

def clone(df_cp):
    pred = df_cp['predict'].values
    return pred

def fprophet_dataset_predict(df_res,delimiter):
    pred = np.where( (abs(df_res['value'].values-df_res['yhat'].values) > delimiter), 1, 0 )
    return pred



def load_pickle_and_predict(KPI, df_test, df_res, df_andrei, df_donut=None):
    # Pickle models in /PICKLE folder
    # Values column accessed with 'df_test.value', you may need it for your predictions
    
    print('\n'+KPI+ ':')
    pred = [0 for i in range(df_test.shape[0])]

    numpy_dir = 'SCORES/predictions_andrei_v15/'
    # numpy_dir = 'SCORES/merged_npy/'




    if KPI == '02e99bd4f6cfb33f':
        # Make predictions
        # pred = fprophet_dataset_predict(df_res,1)

        pred = fprophet_dataset_predict(df_res,0.75)
        pred = naive_test(pred,df_test.value.head(30*1440).values,threshold=0.87)
        # pred = fprophet_dataset_predict(df_res,0.7)



    elif KPI == '046ec29ddf80d62e':
        # Load PICKLE model here
        pred = np.load(numpy_dir + KPI + '_numpy.npy')

        # Make predictions
        # pred = naive_classifier(df_test, down_limit=-1.6)

    elif KPI == '07927a9a18fa19ae':
        # Load PICKLE model here

        # Make predictions
        pred = np.load(numpy_dir + KPI + '_numpy.npy')
        pred = naive_classifier(df_test, down_limit=-1.46)


        # pred = naive_classifier(df_test, down_limit=-1.44)
        # pred = naive_classifier(df_test, down_limit=-1.475)



    elif KPI == '09513ae3e75778a3':
        # Make predictions
        # pred = clone(df_andrei)
        
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == '18fbb1d5a5dc099d':
        # Load PICKLE model here

        # Make predictions
        # pred = naive_classifier(df_test, up_limit=12.7,down_limit=1.3)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')
        pred = naive_adjuster(pred, df_test, 25, up=True)



    elif KPI == '1c35dbf57f55f5e4':
        # Load PICKLE model here

        # Make predictions
        pred = np.load(numpy_dir + KPI + '_numpy.npy')

        # pred = naive_classifier(df_test, up_limit=1800, down_limit=1000)

        # pred = clone(df_andrei)
        # print(pred.shape)
        # pred = naive_adjuster(pred, df_test, 1800, True)
        # pred = naive_adjuster(pred, df_test, 1000, False)

    elif KPI == '40e25005ff8992bd':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,490)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == '54e8a140f6237526':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,2)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == '71595dd7171f4540':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,540)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == '769894baefea4e9e':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,0.045)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == '76f4550c43334374':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,0.037)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == '7c189dd36f048a6c':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,700)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == '88cf3a776ba00e7c':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,0.038)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')

        # pred = clone(df_andrei)

    elif KPI == '8a20c229e9860d0c':
        # Load PICKLE model here

        # Make predictions
        print(df_res.shape)
        pred = fprophet_dataset_predict(df_res,0.06)

    elif KPI == '8bef9af9a922e0b3':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,650)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == '8c892e5525f3e491':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,2100)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')

    elif KPI == '9bd90500bfd11edb':
        # Load PICKLE model here

        # Make predictions
        pred = fprophet_dataset_predict(df_res,16)
        pred = naive_adjuster(pred,df_test,6.0,False)

        # pred = np.load(numpy_dir + KPI + '_numpy.npy')

    elif KPI == '9ee5879409dccef9':
        # Load PICKLE model here

        # Make predictions
        # pred = clone(df_andrei)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    # pred = np.zeros(df_test.shape[0])
    elif KPI == 'a40b1df87e3f1c87':
        # Load PICKLE model here ('/PICKLE/model_name')

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,660)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == 'a5bf5d65261d859a':
        # Load PICKLE model here

        # Make predictions
        pred = naive_classifier_extra(df_test, limit=4.0)

    elif KPI == 'affb01ca2b4f0b45':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,620)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == 'b3b2e6d1a791d63a':

        # Make predictions
        pred = fprophet_dataset_predict(df_res,2.2)

    elif KPI == 'c58bfcbacb2822d1':    # df_status = pd.read_csv('status.csv', columns=['FileName','Time'])
        # Load PICKLE model here

        # Make predictions
        # pred = clone(df_andrei)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == 'cff6d3c01e6a6bfa':
        # Load PICKLE model here

        # Make predictions
        # pred = fprophet_dataset_predict(df_res,460)
        pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == 'da403e4e3f87c9e0':
        # Load PICKLE model here

        # Make predictions
        pred = fprophet_dataset_predict(df_res,1.4)
        
        # pred = naive_adjuster(pred, df_test, 1.47, False)

        pred = naive_adjuster(pred, df_test, 0.91, False)

        # pred = np.load(numpy_dir + KPI + '_numpy.npy')


    elif KPI == 'e0770391decc44ce':
        # Load PICKLE model here

        # Make predictions
        pred = fprophet_dataset_predict(df_res,2750)

    
    return pred


# Loads pickle models and makes predictions
def body():

    df = pd.read_csv('test/test.csv',index_col=['timestamp','KPI ID'])
    df['predict'] = np.zeros(df.shape[0])

    # df_status = pd.read_csv('status.csv', columns=['FileName','Time'])

    df_KPI = pd.read_csv('KPI/KPI.csv')
    KPI_arr = np.array(df_KPI.KPI)

    test_dir = 'test/KPI/test_'
    res_dir = 'DATASETS/fbprophet/fb_test_'
    andrei_dir = 'DATASETS/andrei/andrei_'

    for KPI in KPI_arr:
        df_test = pd.read_csv(test_dir + KPI + '.csv', index_col='timestamp')
        df_res  = pd.read_csv(res_dir + KPI + '.csv', index_col='ds')
        df_andrei = pd.read_csv(andrei_dir + KPI + '.csv', index_col='timestamp')

        predictions = load_pickle_and_predict(KPI,df_test,df_res,df_andrei)

        np.save('SCORES/fbprophet/albert_' + KPI + '.npy',predictions)

        update_score(df, df_test, predictions)

    df = df.reorder_levels(['KPI ID','timestamp'])
    print(df.groupby(df.predict).count())
    df = df.drop(['value'], axis=1)
    df.to_csv('SCORES/score19-0.csv')

if __name__ == "__main__":
    body()
