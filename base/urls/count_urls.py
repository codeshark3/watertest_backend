from django.urls import path
from base.views import test_views as views

urlpatterns = [
  
 
    
    path('', views.getCount, name="count"),

    ] 