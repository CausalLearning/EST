import os
import torch
import time
import numpy as np

def save_checkpoint(state,opt):
    expr_dir = os.path.join(opt.checkpoints_dir,opt.name)

    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)

    epoch = state['epoch']
    save_dir = os.path.join(expr_dir,str(epoch)+'_'+str(round(float(state['prec1']), 4)))
    torch.save(state, save_dir)
    print(save_dir)

def cate2label(dataset_name):
    cate2label = {'MMI': {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise',
                          'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5},
                  'BU3D': {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise',
                           'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5},
                  'AFEW': {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise',
                           'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6},
                  'DFEW':{0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise',
              'Angry': 0,'Disgust': 1,'Fear': 2,'Happy': 3,'Neutral': 4,'Sad': 5,'Surprise': 6},
              'Group': {0: 'sadness', 'sadness': 0, 1: 'happiness', 'happiness': 1, 2: 'helplessness', 'helplessness': 2, 3: 'anxiety', 'anxiety': 3, 4: 'disgust', 'disgust': 4, 5: 'contempt', 'contempt': 5, 6: 'disappointment', 'disappointment': 6, 7: 'surprise', 'surprise': 7, 8: 'fear', 'fear': 8, 9: 'neutral', 'neutral': 9, 10: 'anger', 'anger': 10},
              'Group_multi':{'helplessness_disappointment': 0, 0: 'helplessness_disappointment', 'disgust_helplessness': 1, 1: 'disgust_helplessness', 'fear_surprise': 2, 2: 'fear_surprise', 'anger_surprise': 3, 3: 'anger_surprise', 'disgust': 4, 4: 'disgust', 'anger_sadness': 5, 5: 'anger_sadness', 'disgust_contempt': 6, 6: 'disgust_contempt', 'neutral': 7, 7: 'neutral', 'anxiety_helplessness': 8, 8: 'anxiety_helplessness', 'fear_anxiety': 9, 9: 'fear_anxiety', 'surprise': 10, 10: 'surprise', 'disgust_anxiety': 11, 11: 'disgust_anxiety', 'anger_disgust_contempt': 12, 12: 'anger_disgust_contempt', 'fear_sadness': 13, 13: 'fear_sadness', 'sadness_disappointment': 14, 14: 'sadness_disappointment', 'anxiety': 15, 15: 'anxiety', 'fear': 16, 16: 'fear', 'happiness_contempt': 17, 17: 'happiness_contempt', 'sadness_helplessness_disappointment': 18, 18: 'sadness_helplessness_disappointment', 'disgust_anxiety_helplessness': 19, 19: 'disgust_anxiety_helplessness', 'anger': 20, 20: 'anger', 'disgust_disappointment': 21, 21: 'disgust_disappointment', 'surprise_anxiety': 22, 22: 'surprise_anxiety', 'happiness': 23, 23: 'happiness', 'sadness': 24, 24: 'sadness', 'anger_disgust': 25, 25: 'anger_disgust', 'anger_disgust_anxiety': 26, 26: 'anger_disgust_anxiety', 'fear_surprise_anxiety': 27, 27: 'fear_surprise_anxiety', 'helplessness': 28, 28: 'helplessness', 'sadness_anxiety': 29, 29: 'sadness_anxiety', 'disgust_surprise': 30, 30: 'disgust_surprise', 'disappointment': 31, 31: 'disappointment', 'sadness_anxiety_helplessness': 32, 32: 'sadness_anxiety_helplessness', 'disgust_sadness': 33, 33: 'disgust_sadness', 'anxiety_helplessness_disappointment': 34, 34: 'anxiety_helplessness_disappointment', 'disgust_helplessness_disappointment': 35, 35: 'disgust_helplessness_disappointment', 'happiness_surprise': 36, 36: 'happiness_surprise', 'sadness_surprise': 37, 37: 'sadness_surprise', 'anger_helplessness': 38, 38: 'anger_helplessness', 'fear_sadness_anxiety': 39, 39: 'fear_sadness_anxiety', 'contempt': 40, 40: 'contempt', 'anger_anxiety': 41, 41: 'anger_anxiety', 'sadness_helplessness': 42, 42: 'sadness_helplessness'}
              }

    return cate2label[dataset_name]

def mkdirs(root):
    if not os.path.exists(root):
        os.makedirs(root)

def get_time():
    return str(time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime()))

def draw_weight(weight,file_path):
    np.save(file_path,weight)

