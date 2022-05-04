
from django.db import models
from django.contrib.auth.models import User
import cv2
from PIL import Image
import numpy as np
from django.conf import settings
from pathlib import Path
import os
from .utils import get_oncho_images
from .utils import get_schisto_images
from .utils import get_lf_images
from .utils import get_helminths_images

from django.core.files.base import ContentFile
from io import BytesIO

# Create your models here.
class Test(models.Model):
    _id = models.AutoField(primary_key=True, editable=False)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    name = models.CharField(max_length=200, null=True, blank=True)
    age = models.IntegerField( null=True, blank=True)
    sex = models.CharField(max_length=200, null=True, blank=True)
    location = models.CharField(max_length=200, null=True, blank=True)
    created_at = models.DateField(auto_now=True)
    # oncho =models.IntegerField( null=True, blank=True)
    # schisto =models.IntegerField( null=True, blank=True)
    # lf = models.IntegerField( null=True, blank=True)
    # helminths = models.IntegerField( null=True, blank=True)
   
    oncho =models.CharField(max_length=200, null=True, blank=True)
    schisto =models.CharField(max_length=200, null=True, blank=True)
    lf = models.CharField(max_length=200, null=True, blank=True)
    helminths = models.CharField(max_length=200, null=True, blank=True)
   
    
    onchoImage = models.ImageField(null=True,blank=True)
    schistoImage =  models.ImageField(null=True,blank=True)
    lfImage = models.ImageField(null=True,blank=True)
    helminthsImage =  models.ImageField(null=True,blank=True)
  
    # oncho =models.CharField(max_length=200, null=True, blank=True)
    # schisto =models.CharField(max_length=200, null=True, blank=True)
    # lf = models.CharField(max_length=200, null=True, blank=True)
    # helminths = models.CharField(max_length=200, null=True, blank=True)
    def __str__(self):
        return self.name

    def getCount(self):
        ocount = self.oncho.count()




    def save(self, *args, **kwargs):
          
        
        
        #open image
        onc_img = Image.open(self.onchoImage)
        sch_img = Image.open(self.schistoImage)
        lf_img = Image.open(self.lfImage)
        hel_img = Image.open(self.helminthsImage)
        
     

        # convert image to array and process
        onc_img = np.array(onc_img)
        sch_img = np.array(sch_img)
        lf_img = np.array(lf_img)
        hel_img = np.array(hel_img)

        oncho_result = get_oncho_images(onc_img)
        sch_result = get_schisto_images(sch_img)
        lf_result = get_lf_images(lf_img)
        hel_result = get_helminths_images(hel_img)


        if oncho_result < 2 :
       
            print('Negative')
            self.oncho = 0
        elif oncho_result == 2:
    
            self.oncho = 1
            print('Positive')
        elif oncho_result> 2:
            self.oncho = 0
            print('Not Valid')
       #schisto result
        if sch_result < 2 :
        
            print('Negative')
            self.schisto = 0
        elif sch_result == 2:

            self.schisto = 1
            print('Positive')
        elif sch_result> 2:
            self.schisto = 0
            print('Not Valid')
       
        #lf result
        if lf_result < 2 :
       
            print('Negative')
            self.lf = 0
        elif lf_result == 2:
            self.lf = 1
            print('Positive')
        elif lf_result> 2:
            self.lf =0
            print('Not Valid')
       
       #helminths
        if hel_result < 2 :
       
            print('Negative')
            self.helminths = 0
        elif hel_result == 2:
    
            self.helminths = 1
            print('Positive')
        elif hel_result> 2:
            self.helminths = "Not Valid"
            print('Not Valid')
       

      
     
        
        super().save(*args, **kwargs)  # Call the "real" save() method.
        # do_something_else()
  





# from django.contrib.staticfiles.storage import staticfiles_storage
# from django.db import models
# from django.contrib.auth.models import User
# import cv2
# from PIL import Image
# import numpy as np
# from django.conf import settings
# from django.core.files.storage import get_storage_class
# from .utils import get_images
# from io import BytesIO
# # Create your models here.
# class Test(models.Model):
#     user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
#     name = models.CharField(max_length=200, null=True, blank=True)
#     age = models.IntegerField( null=True, blank=True)
#     sex = models.CharField(max_length=200, null=True, blank=True)
#     location = models.CharField(max_length=200, null=True, blank=True)
#     onchoImage = models.ImageField(upload_to='images')
#     pil_img = Image.open(onchoImage)
#     # img_name = pil_img.filename
#     # print('image name is' +img_name)

#     # convert image to array and process
#     cv_img = np.array(pil_img)
#     img = get_images(cv_img)

#     img = get_images(cv_img)

   
  
#     if img < 2 :
#       oncho =models.CharField(max_length=200, default='Negative')
#     elif img== 2:
#         oncho =models.CharField(max_length=200,default='Positive')
#     elif img > 2:
#         oncho =models.CharField(max_length=200,default='Not Valid')
    
#     print('number of lines in test is ', img)
#     schisto =models.CharField(max_length=200, null=True, blank=True)
#     lf = models.CharField(max_length=200, null=True, blank=True)
#     helminths = models.CharField(max_length=200, null=True, blank=True)
#     enteredAt = models.DateTimeField(auto_now_add=True)
   
#     schistoImage =  models.ImageField(null=True,blank=True)
#     lfImage = models.ImageField(null=True,blank=True)
#     helminthsImage =  models.ImageField(null=True,blank=True)
  
    
#     def __str__(self):
#         return self.name
#     def save(self, *args, **kwargs):
       
#         #open image

#         pil_img = Image.open(self.onchoImage)
#         # img_name = pil_img.filename
#         # print('image name is' +img_name)

#         # convert image to array and process
#         cv_img = np.array(pil_img)
#         img = get_images(cv_img)

#         # #conver back to image
#         # im_pil = Image.fromarray(img)
        

#         # print('number of lines in test is ', len(contours)/2 )

#         # if img== 0:
#         #     oncho_value = "Not Valid 0"
     

#         super().save(*args, **kwargs)  # Call the "real" save() method.
#         # do_something_else()
  


