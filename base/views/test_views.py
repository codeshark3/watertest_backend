from django.shortcuts import render
import itertools
from rest_framework import serializers
from rest_framework.decorators import api_view,permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated,IsAdminUser
from django.core.paginator import Paginator,EmptyPage,Page,PageNotAnInteger

from base.models import Test
from base.serializers import TestSerializer


from django.db.models.functions import TruncMonth
from django.forms.models import model_to_dict
from django.db.models import Sum,Count

import csv
import pandas as pd
import numpy as np 
from sklearn.metrics import mean_squared_error
#from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import datetime
import json


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getTests(request):
    query = request.query_params.get('keyword')
    if query == None:
        query = ''

    tests = Test.objects.filter(name__icontains=query).order_by('_id')
    page = request.query_params.get('page')
    paginator = Paginator(tests,40)

    try:
        tests = paginator.page(page)
    except PageNotAnInteger:
        tests = paginator.page(1)
    except EmptyPage:
        tests = paginator.page(paginator.num_pages)

    if page == None:
        page = 1
    page = int(page)


    serializer = TestSerializer(tests,many=True)
    return Response({'tests':serializer.data,'page':page,'pages':paginator.num_pages}) 


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getTestsMobile(request):
  
    tests = Test.objects.all().order_by('_id')

    serializer = TestSerializer(tests,many=True)
    return Response(serializer.data) 



@api_view(["POST"])
@permission_classes([IsAuthenticated])
def createTest(request):
    data = request.data
   
    test = Test.objects.create(
        name=data['name'],
        age=data['age'],
        sex=data['sex'],
        location=data['location'],
        onchoImage = request.FILES.get('onchoImage'),     
        schistoImage = request.FILES.get('schistoImage'),
        lfImage = data['lfImage'],
        helminthsImage = data['helminthsImage'],
    )
    serializer = TestSerializer(test,many=False)
    return Response(serializer.data)
   
# @api_view("PUT")
# def processImage(request):
#     data = request.data  
   
   
 
  
      
#     # schistoImage = data['schistoImage'],
#     # lfImage = data['lfImage'],
#     # helminthsImage = data['helminthsImage'],
#     test = Test.objects.get(id      
#     )  

#     serializer = TestSerializer(test, )
#     return Response(serializer.data)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getTest(request,pk):
    test = Test.objects.get(_id=pk)
    serializer = TestSerializer(test, many=False)
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getCount(request):
    oTests =Test.objects.count()
    sTests=Test.objects.count()
    lTests=Test.objects.count()
    hTests=Test.objects.count()

    opTests=Test.objects.filter(oncho='1').count()
    spTests=Test.objects.filter(schisto='1').count()
    lpTests=Test.objects.filter(lf='1').count()
    hpTests=Test.objects.filter(helminths='1').count()

    onTests=oTests-opTests
    snTests=sTests-spTests
    lnTests=lTests-lpTests
    hnTests=hTests-hpTests

    countList = {
        "oTests":Test.objects.count(),
        "sTests":Test.objects.count(),
        "lTests":Test.objects.count(),
        "hTests":Test.objects.count(),
        "opTests":Test.objects.filter(oncho='1').count(),
        "spTests":Test.objects.filter(schisto='1').count(), 
        "lpTests":Test.objects.filter(lf='1').count(), 
        "hpTests":Test.objects.filter(helminths='1').count(),
        "onTests":oTests-opTests,
        "snTests":sTests-spTests,
        "lnTests":lTests-lpTests,
        "hnTests":hTests-hpTests
         }
         
    # serializer = TestSerializer(count, many=False)
    return Response(countList)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getCharts(request):
    df=pd.read_csv('sample.csv', )
    series_value = df.values
    # df.describe()
    mean = df.rolling(window=10).mean()
    df_train = df[0:104]
    df_test = df[104:144]


    #2,0,3, 3,0,2
    model =ARIMA(df_train['positive'], order=(2,0,3))
    model_fit = model.fit()
    # model_fit.aic
    forecast = model_fit.forecast(steps= 40)
    test_values= df_test['positive']
    df_test['date']= pd.to_datetime(df_test['date'])
    test_dates = df_test['date']
    #me = np.sqrt(mean_squared_error(test_values,forecast))
    # df_forecast = pd.DataFrame('y_test':y_test,'y_forecast':y_forecast})
    df_final = pd.DataFrame({'dates':test_dates,'forecast':forecast,'test_values':test_values})
    df_final.to_csv('results.csv', index=False)

    filehandle = open('results.csv', 'r')
    charts = csv.DictReader(filehandle,delimiter=",")
    
    
    return Response( charts)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getPrediction(request):
    df=pd.read_csv('sample.csv', )
    series_value = df.values
    # df.describe()
   # mean = df.rolling(window=10).mean()
    df_train = df[0:104]
    # df_test = df[104:144]


    df_train['date']= pd.to_datetime(df_train['date'])
    train_dates = df_train['date']
    train_values= df_train['positive']

    df_final = pd.DataFrame({'dates':train_dates,'train_values':train_values})
    df_final.to_csv('train.csv', index=False)

    filehandle = open('train.csv', 'r')
    reader = csv.DictReader(filehandle,delimiter=",")
    # df='new_test2.csv'
    # filehandle = open('new_test2.csv', 'r')
    # reader = csv.DictReader(filehandle)
       
    # df.info()
    # convert the  Date to datetime
    # df['date'] = pd.to_datetime(df['date'])
    # # df.sort_values(by=['date'])
    # # df =df.drop(columns=['negative'])

    # # add a column for Year
    # df['year'] = df['date'].dt.year
    # # df.set_index('year')
    # df['time'] = np.arange(1,101,1)
    # df['month_name'] = df['date'].dt.month_name()
    # df_train = df.head(80)
    # df_test = df.loc[80:]
    # x_train = df_train[['time']].values
    # y_train= df_train['positive'].values

    # x_test = df_test[['time']].values
    # y_test= df_test['positive'].values

    # model = LinearRegression()
    # model.fit(x_train,y_train)
    # y_value = model.predict(x_train)
    # y_forecast = model.predict(x_test)

    # years = np.array(df_train['year'])

    

    # result = {
    # "y-value":   np.array(y_value),
    # "y_forecast":  y_forecast,
    # "years": np.unique(df['date'])
    # }
    # result = json.dumps(reader, indent=4)
    #     df=pd.read_csv('sample.csv')

    # # convert the  Date to datetime
    # df['date'] = pd.to_datetime(df['date'])

    # # df =df.drop(columns=['negative'])

    # # add a column for Year
    # df['year'] = df['date'].dt.year
    # # df.set_index('year')
    # df['time'] = np.arange(1,145,1)
    # df['month_name'] = df['date'].dt.month_name()
    # df_train = df.head(100)
    # df_test = df.loc[100:]
    # x_train = df_train[['time']].values
    # y_train= df_train['positive'].values

    # x_test = df_test[['time']].values
    # y_test= df_test['positive'].values.tolist()

    # model = LinearRegression()
    # model.fit(x_train,y_train)
    # y_value = model.predict(x_train)
    # y_forecast = model.predict(x_test).tolist()


    # train_dates = np.array(df_train['year']).tolist()
    # test_dates = np.array(df_test['year']).tolist()
    
   

    # df_forecast = pd.DataFrame({'dates':test_dates,'y_test':y_test,'y_forecast':y_forecast})
    # df_final = pd.DataFrame({'train_dates': train_dates,'y_train': y_train, 'y_value': y_value})
    # df_final.to_csv('results.csv', index=False)

    # filehandle = open('results.csv', 'r')
    # charts = csv.DictReader(filehandle,delimiter=",")
    return Response(reader)
