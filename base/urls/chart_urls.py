from django.urls import path
from base.views import test_views as views

urlpatterns = [
  
 
    
   
 path('', views.getCharts, name="charts")
    ] 