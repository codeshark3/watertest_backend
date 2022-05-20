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
from django.http import HttpResponse

from django.db.models.functions import TruncMonth
from django.forms.models import model_to_dict
from django.db.models import Avg, Count, Min, Sum

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

# Test.objects.annotate(month = ExtractMonth('created_at')).values('month').annotate(c=Count('_id')) 

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
def getDataTable(request):
    oreader = open('oncho_results.csv', 'r')
    sreader = open('schisto_results.csv', 'r')
    lreader = open('lf_results.csv', 'r')
    hreader = open('helminths_results.csv', 'r')

    odatatable = csv.DictReader( oreader,delimiter=",")
    sdatatable = csv.DictReader( sreader,delimiter=",")
    ldatatable = csv.DictReader( lreader,delimiter=",")
    hdatatable = csv.DictReader( hreader,delimiter=",")
    # oncho_df = pd.read_csv('oncho.csv')
    # schisto_df = pd.read_csv('schisto.csv')
    # lf_df = pd.read_csv('lf.csv')
    # helminths_df = pd.read_csv('helminths.csv')


    # oncho_df = oncho_df.dropna()
    # schisto_df = schisto_df.dropna()
    # lf_df = lf_df.dropna()
    # lf_df =lf_df.dropna()

    # oncho_df['Oncho']= oncho_df['Oncho'].astype(str)
    # schisto_df['Schisto']= schisto_df['Schisto'].astype(str)
    # lf_df['Lf']=lf_df['Lf'].astype(str)
    # helminths_df['Helminths']= helminths_df['Helminths'].astype(str)

    # oncho_df['Created_At'] = pd.to_datetime(oncho_df['Created_At'])
    # schisto_df['Created_At'] = pd.to_datetime(schisto_df['Created_At'])
    # lf_df['Created_At'] = pd.to_datetime(lf_df['Created_At'])
    # helminths_df['Created_At'] = pd.to_datetime(helminths_df['Created_At'])

    # p_oncho_list=['Positive']
    # n_oncho_list = ['Negative']

    # p_schisto_list=['Positive']
    # n_schisto_list = ['Negative']

    # p_lf_list=['Positive']
    # n_lf_list = ['Negative']

    # p_helminths_list=['Positive']
    # n_helminths_list = ['Negative']

    # onchop_df = oncho_df[oncho_df.Oncho.isin(p_oncho_list)]
    # onchop_df = onchop_df['Oncho'].groupby(oncho_df['Created_At']).count()

    # onchon_df = oncho_df[oncho_df.Oncho.isin(n_oncho_list)]
    # onchon_df = onchon_df['Oncho'].groupby(oncho_df['Created_At']).count()




    # ###################
    # schistop_df = schisto_df[schisto_df.Schisto.isin(p_schisto_list)]
    # schistop_df = schistop_df['Schisto'].groupby(schisto_df['Created_At']).count()

    # schiston_df = schisto_df[schisto_df.Schisto.isin(n_schisto_list)]
    # schiston_df = schiston_df['Schisto'].groupby(schisto_df['Created_At']).count()


    # #############3

    # lfp_df = lf_df[lf_df.Lf.isin(p_lf_list)]
    # lfp_df = lfp_df['Lf'].groupby(lf_df['Created_At']).count()

    # lfn_df = lf_df[lf_df.Lf.isin(n_lf_list)]
    # lfn_df = lfn_df['Lf'].groupby(lf_df['Created_At']).count()

    # ##############3
    # helminthsp_df = helminths_df[helminths_df.Helminths.isin(p_helminths_list)]
    # helminthsp_df = helminthsp_df['Helminths'].groupby(helminths_df['Created_At']).count()

    # helminthsn_df = helminths_df[helminths_df.Helminths.isin(n_helminths_list)]
    # helminthsn_df = helminthsn_df['Helminths'].groupby(helminths_df['Created_At']).count()

    # ########### Create new dataframes with dates and values
    # newopdf =pd.DataFrame({'date':onchop_df.index, 'positive':onchop_df.values})
    # newondf =pd.DataFrame({'date':onchon_df.index, 'negative':onchon_df.values})

    # newspdf =pd.DataFrame({'date':schistop_df.index, 'positive':schistop_df.values})
    # newsndf =pd.DataFrame({'date':schiston_df.index, 'negative':schiston_df.values})

    # newlpdf =pd.DataFrame({'date':lfp_df.index, 'positive':lfp_df.values})
    # newlndf =pd.DataFrame({'date':lfn_df.index, 'negative':lfn_df.values})

    # newhpdf =pd.DataFrame({'date':helminthsp_df.index, 'positive':helminthsp_df.values})
    # newhndf =pd.DataFrame({'date':helminthsn_df.index, 'negative':helminthsn_df.values})

    # ofinal = pd.merge(newopdf, newondf, on = "date", how = "outer")
    # ofinal = ofinal.fillna(0)

    # ofinal['sum']= ofinal.sum(axis=1)
    # ofinal=ofinal.append(ofinal[['positive','negative']].sum(),ignore_index=True).fillna('')
    # ############

    # sfinal = pd.merge(newspdf, newsndf, on = "date", how = "outer")
    # sfinal = sfinal.fillna(0)

    # sfinal['sum']= sfinal.sum(axis=1)
    # sfinal=sfinal.append(sfinal[['positive','negative']].sum(),ignore_index=True).fillna('')
    # ####################
    # lfinal = pd.merge(newlpdf, newlndf, on = "date", how = "outer")
    # lfinal = lfinal.fillna(0)

    # lfinal['sum']= lfinal.sum(axis=1)
    # lfinal=lfinal.append(lfinal[['positive','negative']].sum(),ignore_index=True).fillna('')

    # ##################
    # hfinal = pd.merge(newhpdf, newhndf, on = "date", how = "outer")
    # hfinal = hfinal.fillna(0)

    # hfinal['sum']= hfinal.sum(axis=1)
    # hfinal=hfinal.append(hfinal[['positive','negative']].sum(),ignore_index=True).fillna('')

    # ofinal.to_csv('oncho_results.csv', index=False)
    # sfinal.to_csv('schisto_results.csv', index=False)
    # lfinal.to_csv('lf_results.csv', index=False)

    # hfinal.to_csv('helminths_results.csv', index=False)
    
    # response = HttpResponse(content_type='text/csv')
    # with open('helminths.csv', mode='w') as dataFile:     
    #     writer = csv.writer(dataFile, )
    #     writer.writerow([ 'Created_At','Helminths'])

    #     for test in Test.objects.all().values_list( 'created_at','helminths', ):
    #         writer.writerow(test)
        # writer.writerow(['Name','Age','Sex', 'Location', 'Created_At','Oncho', 'Schisto','Lf', 'Helminths' ])

        #         for test in Test.objects.all().values_list('name','age','sex', 'location', 'created_at','oncho', 'schisto','lf', 'helminths' ):
        #             writer.writerow(test)

    #     # response['Content-Disposition'] = 'attachment; filename="dataTable.csv"'

    
    return Response({'odatatable':odatatable,'sdatatable':sdatatable,'ldatatable':ldatatable, "hdatatable":hdatatable})




@api_view(["GET"])
@permission_classes([IsAuthenticated])
def getCount(request):
    oTests =Test.objects.count()
    sTests=Test.objects.count()
    lTests=Test.objects.count()
    hTests=Test.objects.count()

  
    opTests=Test.objects.filter(oncho="Positive").count()
    spTests=Test.objects.filter(schisto="Positive").count()
    lpTests=Test.objects.filter(lf="Positive").count()
    hpTests=Test.objects.filter(helminths="Positive").count()
    onTests=oTests-opTests
    snTests=sTests-spTests
    lnTests=lTests-lpTests
    hnTests=hTests-hpTests

    countList = {
        "oTests":oTests,
        "sTests": sTests,
        "lTests":lTests,
        "hTests":hTests,
        "opTests": opTests,
        "spTests":spTests,
        "lpTests":lpTests,
        "hpTests":hpTests,
        "onTests": onTests,
        "snTests":snTests,
        "lnTests":lnTests,
        "hnTests":hnTests
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
    start_date = datetime.datetime(1959, 5, 24)
    # model_fit.aic
    forecast = model_fit.forecast(steps= 50)
    test_values= df_test['positive']
    df_test['date']= pd.to_datetime(df_test['date'])
    test_dates = df_test['date']
    res = [start_date + datetime.timedelta(days=idx) for idx in range(10)]
    res = pd.Series(res)
    #me = np.sqrt(mean_squared_error(test_values,forecast))
    # df_forecast = pd.DataFrame('y_test':y_test,'y_forecast':y_forecast})
    # df_final = pd.DataFrame({'dates':test_dates,'forecast':forecast,'test_values':test_values})

    dates = test_dates.append(res, ignore_index = True).tolist()

    df_final = pd.DataFrame({'dates':dates,'forecast':forecast,'test_values':test_values})
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
    df_train = df[104:144]
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
