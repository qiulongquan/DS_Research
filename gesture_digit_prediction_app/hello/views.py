from .models import Img
from django.shortcuts import render
from tensorflow.keras.preprocessing import image
from django.utils import timezone
from django.utils.timezone import localtime
import matplotlib.pyplot as plt
import numpy as np
import os
# shutil是文件 文件夹操作相关方法的package
import shutil
import tensorflow as tf
import keras
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras import models
from keras.models import load_model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# https://docs.djangoproject.com/ja/3.0/topics/http/file-uploads/

dirs='/workspace/python/temp_pic'
dirs_0='/workspace/python/temp_pic/patterndir'
BATCH_SIZE=32
IMG_SIZE = (224, 224)

def index(request):
    now = localtime(timezone.now())
    print("qiulongquan_now={}".format(now))
    context = {'now': now, 'qiu': "customization"}
    return render(request, 'hello/index.html', context)


def create_temp_pic_dir(img_path):
    if not os.path.exists(dirs):
        os.mkdir(dirs)
        os.mkdir(dirs_0)
    else:
#     强制删除dirs文件夹内的所有内容并删除dirs文件夹
        shutil.rmtree(dirs) 
        os.mkdir(dirs)
        os.mkdir(dirs_0)
    if img_path is not None and os.path.splitext(img_path.split('/')[-1])[-1] == ".jpg":
        src=img_path
        dst=os.path.join(dirs_0,img_path.split('/')[-1])
        shutil.copyfile(src,dst)
        print("copy done={}".format(dst))
    else:
        print("no file copy")


def preprocessing(dir_path):
    print("qiu_dir_path",dir_path)
    picture_data = image_dataset_from_directory(dir_path,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    picture_data = picture_data.prefetch(buffer_size=AUTOTUNE)
    image, _ = next(iter(picture_data))
    return image


def prediction_process(img_paths):
    print(tf.__version__)
    classes = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine']
    
    model_path = os.path.join("model", 'efficientnet_unfreeze_model.h5')
#     model_path = os.path.join("model", 'mobilenet_v2_model.h5')
#     model_path = os.path.join("model", 'sign_language_vgg16.h5')
    model = load_model(model_path)
    print(model.summary())
    prediction_results = []
    result_set = []
    
    for img_path in img_paths:
        if img_path is not None:
            create_temp_pic_dir(img_path)
            prepro_image=preprocessing(dirs)
            pred = model.predict(prepro_image)
            Predict_value = pred[0].argmax()
            print("qiulongquan_predict={}".format(Predict_value))
            prediction_results.append(Predict_value)
            top_indices = pred[0].argsort()[-5:][::-1]
            result_five = [(classes[i], pred[0][i]) for i in top_indices]
            print("result_five type={}".format(type(result_five)))
            result_set.append(result_five)
            for x in result_five:
                print(x)
        elif img_path is None:
            prediction_results.append("")
            result_set.append("")
            
    return prediction_results, result_set


def prediction(request):
    print("prediction image")
    if request.method == 'POST':
        checkbox_values = request.POST.getlist('checks[]')
        print("qiulongquan_POST_checkbox[]={}".format(checkbox_values))
        imgs = Img.objects.all()
        img_paths = []
        for i, img in enumerate(imgs):
            if str(i) in checkbox_values:
                print("img_path={}".format(os.path.join("media", img.img.name)))
                img_paths.append(os.path.join("media", img.img.name))
            else:
                img_paths.append(None)
        prediction_results, result_set = prediction_process(img_paths)
        content = {
            'imgs': imgs,
            'port': request.META['SERVER_PORT'],
            'host': request.get_host(),
            'img_paths': img_paths,
            'prediction_results': prediction_results,
            'result_set': result_set,
        }
        return render(request, 'hello/showing.html', content)


def handle_uploaded_file(f, img_path):
    with open(img_path, 'wb') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
        destination.close()


def uploadImg(request):
    """
    :param request:
    :return:
    """
    print("upload files and then show files")
    if request.method == 'POST':
        img_files = request.FILES.getlist("img")
        for img_file in img_files:
            name = img_file.name
            img_path = os.path.join("media/file", name)
            if os.path.exists(img_path.encode('utf-8')):
                print(str(img_path.encode('utf-8')) + "  exists.")
                continue
            else:
                print(str(img_path) + "  no exists.")
                handle_uploaded_file(img_file, img_path)
                Img(img=os.path.join("file", name), name=name).save()
                print("{} upload done.".format(name))
    return render(request, 'hello/uploading.html')
    # return HttpResponsePermanentRedirect("/s/" + code + "/")


# from django.shortcuts import render
# from .models import Img
#
#
# def uploadImg(request):
#     """
#     :param request:
#     :return:
#     """
#     if request.method == 'POST':
#         new_img = Img(
#             img=request.FILES.get('img'),
#             name=request.FILES.get('img').name
#         )
#         new_img.save()
#     return render(request, 'hello/uploading.html')


def showImg(request):
    """
    :param request:
    :return:
    """
    print("qiulongquan_showImg_start")
    imgs = Img.objects.all()
    content = {
        'imgs': imgs,
        'port': request.META['SERVER_PORT'],
        'host': request.get_host(),
    }
    print("qiulongquan_content={}".format(content))
    for i in imgs:
        print("qiulongquan_url={}".format(i.img.url))
    return render(request, 'hello/showing.html', content)


def deleteImg(request):
    delete_list = request.POST.getlist('checks[]')
    print("qiulongquan_delete_list={}".format(delete_list))
    imgs = Img.objects.all()
    for img in imgs:
        if str(img.id) in delete_list:
            file_path=str(img.img.url)
            if file_path[0:1]=='/':
                file_path=file_path[1:]
            if os.path.exists(os.path.join('media', file_path)):
                os.remove(os.path.join('media', file_path))
                print("%s delete completed" %
                      os.path.join('media', file_path))
                Img.objects.get(id=img.id).delete()
            else:
                print('no such file:%s' %
                      os.path.join('media', file_path))
    print("delete files done.")
    imgs = Img.objects.all()
    content = {
        'imgs': imgs,
        'port': request.META['SERVER_PORT'],
        'host': request.get_host(),
    }
    return render(request, 'hello/showing.html', content)
