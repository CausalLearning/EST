import os
import random
import torch.utils.data as data
from PIL import Image
import torch
import numpy as np

class RandomShuffleDataset(data.Dataset):
    def __init__(self,video_root,video_list,isTrain,rectify_label,opt,transform=None):
        super(RandomShuffleDataset, self).__init__()
        ### param area####################
        self.first_clips_number = 30  # 从前first_clips_number帧中选第一个clip的第一帧
        self.search_clips_number = 75 # 从第一个clip之后往后找search_clips_number帧做剩余clips-1的数据
        self.choose_num = opt.per_snippets  # 每个clip选choose_num张图
        self.each_clips=15 # 每个clip的选取范围
        self.clips = opt.snippets # 多少个clip
        self.next_jump = 10 # 当前位置的下一跳
        self.isTrain = isTrain
        self.test_first = opt.test_first
        self.opt = opt

        #####data path ###################
        self.video_root = video_root
        self.video_list = video_list
        self.rectify_label = rectify_label
        self.transform = transform
        self.permutation = opt.permutation_root
        self.per_nums = opt.permutation_classes
        #############################

        self.video_label = self.read_data(self.video_root,self.video_list,self.rectify_label)
        self.permutation = self.load_per(self.permutation).tolist()

    def load_per(self,path):
        return np.load(path)

    def read_data(self,video_root,video_list,rectify_label):
        video_label_list = []
        #print(video_list)
        # 读取文件，获取所有视频数据
        with open(video_list,'r') as imf:

            for id, line in enumerate(imf):
                video_label = line.strip().split()

                video_name = video_label[0]
                label = rectify_label[video_label[1]]

                video_label_list.append((os.path.join(video_root,video_name),label))

        return video_label_list

    def __getitem__(self, index):
        '''
        :param index:
        :return:
        data = [clip_0,clip_1,clip_2,...,clip_7]
        clip_0 = [torch.tensor_0,torch.tensor_1,...,torch.tensor_4] torch.tensor_0.shape = [3,224,224]
        label = 0 or 1 or 2 or 3 or 4 or 5 ....
        path: 当前视频的路径
        '''
        data_path,label = self.video_label[index]

        frame_path_list = sorted(os.listdir(data_path))

        if self.isTrain:
            first_loc = random.randint(0, self.first_clips_number - 1)
        else:
            first_loc = self.test_first
        # 从first_loc开始往后选each_clips+search_clips_number张图片，each_clips为第一个clip的张数，后面的search_clips_number为后面的clip所需要的图片数
        sub_frames_list = frame_path_list[first_loc:first_loc + self.search_clips_number]

        data_clips = []

        # clip 0~clips-1
        cur_loc = 0
        for i in range(0, self.clips):
            high_range = cur_loc + self.each_clips
            low_range = cur_loc
            frames_tmp = sub_frames_list[low_range:high_range]
            data_clips.append(self.get_image(frames_tmp, data_path))

            cur_loc += self.next_jump

        per_loc = random.randint(0,self.per_nums - 1)
        per = self.permutation[per_loc]

        data_clips_shuffle = self.order_clip(data_clips,per)
        data_clips_normal = self.order_clip(data_clips,[x for x in range(self.opt.snippets)])

        return {'data_normal':data_clips_normal,'label':label,'path':data_path,'data_shuffle':data_clips_shuffle,'per_label':per_loc,'per_shuffle':torch.tensor(per),'per_normal':torch.tensor([x for x in range(self.opt.snippets)])}

    def order_clip(self,data_clips,order):
        clip_list = [torch.stack(data_clips[order[i]],dim=0) for i in range(len(order))]
        return torch.stack(clip_list, dim=0)

    # 读取每个clip的图片数据
    def get_image(self,frames,data_path):
        # 随机采样 choose_num 个图片id
        if self.isTrain:
            indexs = self.sample(self.choose_num,0,self.each_clips-1)
        else:
            indexs = [x for x in range(0,self.each_clips,self.each_clips // self.choose_num)]
        # 读取图片
        result_list = []
        for loc in indexs:
            assert len(frames)> loc, data_path
            img_path = os.path.join(data_path,frames[loc])
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224,224))
            if self.transform is not None:
                img = self.transform(img)
            result_list.append(img)

        return result_list

    def sample(self,num,min_index,max_index):
        s = set()
        while len(s) < num:
            tmp = random.randint(min_index,max_index)
            s.add(tmp)
        return list(s)

    def __len__(self):
        return len(self.video_label)


