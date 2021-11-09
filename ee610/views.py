from django.shortcuts import render
from django.http import HttpResponse
import cv2
import urllib
import numpy as np
from django.views.decorators.csrf import csrf_exempt
import os
from skimage import color

# Create your views here.


def home(request):
    return render(request, 'home.html')


@csrf_exempt
def make_negative(request):
    # Getting the image
    try:
        img = request.FILES.get("file")
        image_name = request.POST['image_name']
        image_name = os.path.basename(image_name)
        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)

    except:        
        image_name = request.POST['image_name']
        image_name = os.path.basename(image_name)
        img = urllib.request.urlopen('http://127.0.0.1:8000/static/' + image_name)
        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)

    cv2.imwrite('./static/'+image_name, img_bgr)
    cv2.imwrite('./staticfiles/'+image_name, img_bgr)
    #If RGB image
    if len(img_bgr.shape) == 3:
        height, width, _ = img_bgr.shape

        for i in range(0, height - 1):
            for j in range(0, width - 1):

                # Get the pixel value and subtract it from 255 to have negative
                pixel = img_bgr[i, j]

                pixel[0] = 255 - pixel[0]

                pixel[1] = 255 - pixel[1]

                pixel[2] = 255 - pixel[2]

                img_bgr[i, j] = pixel

        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)

        return HttpResponse("Done")
    #if gray image
    if len(img_bgr.shape) == 2:
        height, width = img_bgr.shape

        for i in range(0, height - 1):
            for j in range(0, width - 1):

                # Get the pixel value and subtract it from 255 to have negative
                pixel = 255 - img_bgr[i, j]

                img_bgr[i, j] = pixel

        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)

        return HttpResponse("Done")


@csrf_exempt
def hist_equal(request):
    #getting the image
    try:
        img = request.FILES.get("file")
        image_name = request.POST['image_name']
        image_name = os.path.basename(image_name)

        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        

    except:        
        image_name = request.POST['image_name']
        image_name = os.path.basename(image_name)
        img = urllib.request.urlopen('http://127.0.0.1:8000/static/' + image_name)
        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
    
    if len(img_bgr.shape) == 3:
        #converting to hsv
        img_bgr = color.rgb2hsv(img_bgr)
        # getting the intensity channel
        gray_img_bgr = img_bgr[:,:,2]
        gray_img_bgr  = my_histeq2(gray_img_bgr )
        img_bgr[:,:,2] = gray_img_bgr 
        img_bgr = color.hsv2rgb(img_bgr)

        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)

    if len(img_bgr.shape) == 2:
        gray_img_bgr = img_bgr
        gray_img_bgr  = my_histeq(gray_img_bgr )
        img_bgr = gray_img_bgr
        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)

    return HttpResponse("Done")

@csrf_exempt
def log_transform(request):
    #getting the image
    try:
        img = request.FILES.get("file")
        image_name = request.POST['image_name']
        image_name = os.path.basename(image_name)

        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        

    except:        
        image_name = request.POST['image_name']
        image_name = os.path.basename(image_name)
        img = urllib.request.urlopen('http://127.0.0.1:8000/static/' + image_name)
        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        

    if len(img_bgr.shape) == 3:
        #converting to hsv
        img_bgr = color.rgb2hsv(img_bgr)
        #extracting the intensity channel
        gray_img_bgr = img_bgr[:,:,2]
        #writing the log transform formula
        gray_img_bgr = (np.log(gray_img_bgr+1)/(np.log(1+np.max(gray_img_bgr))))*255
        #merging the intensity channel back
        img_bgr[:,:,2] = gray_img_bgr 
        #converting back to rgb
        img_bgr = color.hsv2rgb(img_bgr)
        #saving the image
        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        return HttpResponse("Done")
    
    if len(img_bgr.shape) == 2:
        gray_img_bgr = img_bgr/255
        #writing the log transform formula
        gray_img_bgr = (np.log(gray_img_bgr+1)/(np.log(1+np.max(gray_img_bgr))))*255
        #saving the image
        image_name = image_name + '.jpg'
        img_bgr = gray_img_bgr
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        return HttpResponse("Done")

@csrf_exempt
def gamma_correct(request):
    #getting the image and gamma value
    try:
        img = request.FILES.get("file")
        image_name = request.POST['image_name']
        gamma_value = request.POST['gamma_value']
        image_name = os.path.basename(image_name)

        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        

    except:        
        image_name = request.POST['image_name']
        gamma_value = request.POST['gamma_value']
        image_name = os.path.basename(image_name)
        img = urllib.request.urlopen('http://127.0.0.1:8000/static/' + image_name)
        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)

    gamma_value = float(gamma_value)
    if len(img_bgr.shape) == 3: 
        #converting to hsv  
        img_bgr = color.rgb2hsv(img_bgr)
        #getting the intensity channel
        gray_img_bgr = img_bgr[:,:,2]
        #applying gamma transform formula
        gray_img_bgr = np.array(255*(gray_img_bgr) ** gamma_value, dtype = 'uint8')
        #merging back the intensity channel
        img_bgr[:,:,2] = gray_img_bgr
        #converting back to bgr
        img_bgr = color.hsv2rgb(img_bgr)
        #saving the image
        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        return HttpResponse("Done")

    if len(img_bgr.shape) == 2:
        gray_img_bgr = img_bgr/255
        #applying the gamma transform formula
        gray_img_bgr = np.array(255*(gray_img_bgr) ** gamma_value, dtype = 'uint8')
        img_bgr = gray_img_bgr
        #saving the image
        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        return HttpResponse("Done")

@csrf_exempt
def blur(request):
    #getting the image and blur value
    try:
        img = request.FILES.get("file")
        image_name = request.POST['image_name']
        blur_value = request.POST['blur_value']
        image_name = os.path.basename(image_name)

        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        

    except:        
        image_name = request.POST['image_name']
        blur_value = request.POST['blur_value']
        image_name = os.path.basename(image_name)
        img = urllib.request.urlopen('http://127.0.0.1:8000/static/' + image_name)
        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        

    if len(img_bgr.shape) == 3:
        #converting to hsv
        img_bgr = color.rgb2hsv(img_bgr)
        #extracting the intensity channel
        gray_img_bgr = img_bgr[:,:,2]
        blur_value = int(blur_value)
        #making the blur filter
        mask = np.ones([blur_value, blur_value], dtype = int)
        print(mask)
        mask = mask / (blur_value*blur_value)
        #passing the image and filter into convolution
        gray_img_bgr_2 = my_conv(gray_img_bgr, mask)
        #merging the image back
        img_bgr[:,:,2] = gray_img_bgr_2 
        #converting back to rgb
        img_bgr = color.hsv2rgb(img_bgr)
        #saving the image
        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        return HttpResponse("Done")
    
    if len(img_bgr.shape) == 2:
        gray_img_bgr = img_bgr/255
        blur_value = int(blur_value)
        #making the blur mask
        mask = np.ones([blur_value, blur_value], dtype = int)
        print(mask)
        mask = mask / (blur_value*blur_value)
        #passing it to convolution
        gray_img_bgr_2 = my_conv(gray_img_bgr, mask)
        img_bgr = gray_img_bgr_2
        #saving the image
        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        return HttpResponse("Done")

@csrf_exempt
def sharp(request):
    #getting the image and sharp value
    try:
        img = request.FILES.get("file")
        image_name = request.POST['image_name']
        sharp_value = request.POST['sharp_value']
        image_name = os.path.basename(image_name)

        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        

    except:        
        image_name = request.POST['image_name']
        sharp_value = request.POST['sharp_value']
        image_name = os.path.basename(image_name)
        img = urllib.request.urlopen('http://127.0.0.1:8000/static/' + image_name)
        arr = np.asarray(bytearray(img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, -1)
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)

    if len(img_bgr.shape) == 3:   
        #converting to hsv
        img_bgr = color.rgb2hsv(img_bgr)
        #extracting intensity channe;
        gray_img_bgr = img_bgr[:,:,2]
        sharp_value = float(sharp_value)
        #making sharp mask
        mask = np.array([
                [-1*sharp_value, -1*sharp_value, -1*sharp_value],
                [-1*sharp_value, 1+8*sharp_value, -1*sharp_value],
                [-1*sharp_value, -1*sharp_value, -1*sharp_value]
                ],dtype='float')
        #passing image and mask to conv function
        gray_img_bgr_2 = my_conv(gray_img_bgr, mask)
        #merging the image back
        img_bgr[:,:,2] = gray_img_bgr_2 
        #converted back to rgb
        img_bgr = color.hsv2rgb(img_bgr)
        #saved the image
        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        return HttpResponse("Done")

    if len(img_bgr.shape) == 2:
        gray_img_bgr = img_bgr/255
        sharp_value = float(sharp_value)
        # Making the sharp mask
        mask = np.array([
                [-1*sharp_value, -1*sharp_value, -1*sharp_value],
                [-1*sharp_value, 1+8*sharp_value, -1*sharp_value],
                [-1*sharp_value, -1*sharp_value, -1*sharp_value]
                ],dtype='float')
        #passed the mask and image to conv
        gray_img_bgr_2 = my_conv(gray_img_bgr, mask)
        img_bgr = gray_img_bgr_2
        #saved the image
        image_name = image_name + '.jpg'
        cv2.imwrite('./static/'+image_name, img_bgr)
        cv2.imwrite('./staticfiles/'+image_name, img_bgr)
        return HttpResponse("Done")


def my_histeq2(Image):
    HeqImg = Image
    #extracting width and height
    (R,C) = Image.shape
    Freq = np.zeros((256,1),dtype='float')
    # Getting the number of pixel with intensity v
    for v in np.arange(256, dtype='int'):
        Freq[v] = np.count_nonzero( np.all([ (Image >= v-0.5), (Image < v+0.5) ], axis=0) )
    # subtracting by total number of pixel and getting the cumulative sum
    CDF = np.cumsum(Freq)/(R*C)
    # replacing the equalized values 
    for v in np.arange(256, dtype='int'):
        np.place(HeqImg, np.all([ (Image >= v-0.5), (Image < v+0.5) ], axis=0), CDF[v]*255.0)
    return HeqImg

def my_histeq(Image):
    HeqImg = Image/255
    #extracting width and height
    (R,C) = Image.shape
    Freq = np.zeros((256,1),dtype='float')
    # Getting the number of pixel with intensity v
    for v in np.arange(256, dtype='int'):
        Freq[v] = np.count_nonzero( np.all([ (Image >= v-0.5), (Image < v+0.5) ], axis=0) )
    # subtracting by total number of pixel and getting the cumulative sum
    CDF = np.cumsum(Freq)/(R*C)
    # replacing the equalized values 
    for v in np.arange(256, dtype='int'):
        np.place(HeqImg, np.all([ (Image >= v-0.5), (Image < v+0.5) ], axis=0), CDF[v]*255)
    return HeqImg

def my_conv(Img, filter):
    Ret = Img
    #getting the image shape
    (R, C) = Img.shape
    # getting the filter shape
    (Rf, Cf) = filter.shape
    # zero padding the image
    Img2 = np.append(np.zeros((int((Rf-1)/2),C)), Img, axis=0)
    Img2 = np.append(Img2,np.zeros((int((Rf-1)/2),C)), axis=0)
    Img2 = np.append(np.zeros((R+Rf-1,int((Cf-1)/2))), Img2,axis=1)
    Img2 = np.append(Img2,np.zeros((R+Rf-1,int((Cf-1)/2))),axis=1)
    Img2 = Img2 * 255
    #implementing the standard convolution function
    for i in np.arange( 0 + int((Rf-1)/2),    R + int((Rf-1)/2) ):
        for j in np.arange(0 + int((Cf-1)/2),    C + int((Cf-1)/2) ):
            Ret[i-int((Rf-1)/2), j-int((Cf-1)/2)] = np.sum(filter*Img2[i-int((Rf-1)/2):i+int((Rf-1)/2)+1, j-int((Cf-1)/2): j+int((Cf-1)/2)+1])
            

    return Ret

