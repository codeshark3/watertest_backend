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
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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
   

   labels = [1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]
   values = [10, 20, 30, 40, 50, 60, 70,80,90]
   result = {
    "labels":   labels,
    "values":  values
    }
   return Response(result)