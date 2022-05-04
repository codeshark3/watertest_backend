from django.urls import path
from base.views import test_views as views

urlpatterns = [
  
 
    path('', views.getTests ,name="tests"),
    path('mobile/', views.getTestsMobile ,name="mobile"),
    path('create/', views.createTest ,name="createTest"),
    path('<str:pk>/', views.getTest, name="test"),

   
   
    ] 