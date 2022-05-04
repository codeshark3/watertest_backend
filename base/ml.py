import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def ml_code:
    df=pd.read_csv('new_test2.csv')
    # df.info()
    # convert the  Date to datetime
    df['date'] = pd.to_datetime(df['date'])
    # df.sort_values(by=['date'])
    # df =df.drop(columns=['negative'])

    # add a column for Year
    df['year'] = df['date'].dt.year
    # df.set_index('year')
    df['time'] = np.arange(1,101,1)
    df['month_name'] = df['date'].dt.month_name()
    df_train = df.head(80)
    df_test = df.loc[80:]
    x_train = df_train[['time']].values
    y_train= df_train['positive'].values

    x_test = df_test[['time']].values
    y_test= df_test['positive'].values

    model = LinearRegression()
    model.fit(x_train,y_train)
    y_value = model.predict(x_train)
    y_forecast = model.predict(x_test)



    # #plot data
    # plt.figure(figsize=(12,6))
    # plt.plot(df_train['year'],y_train,'bo')
    # plt.plot(df_train['year'],y_value,'b')

    # plt.plot(df_test['year'],y_test,'o:', color='orange')
    # plt.plot(df_test['year'],y_forecast,'b', color='red')

    #error
    error = y_train-y_value
    se= error.std()
    mean_error = 1.96 * se
    ci_plus =y_forecast + mean_error
    ci_minus = y_forecast - mean_error
