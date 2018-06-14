import numpy as np
from sklearn.cross_validation import train_test_split
import threading
import json
import keras
import cv2

def get_data():
    with open("test.json",'r') as f:
        test_dict = json.load(f)
        test_images = test_dict["images"]
    with open("train.json",'r') as f:
        train_dict = json.load(f)
        annotations = train_dict["annotations"]
    
    total_classes = 997 # test_dict['images']
    total_test = len(test_images)
    test_image_names = []

    for index in range(total_test):
        img_name = "test/"+test_images[index]["file_name"]
        test_image_names.append(img_name)
    test_image_names = np.array(test_image_names)    
    # print("----------------train-----------------------")
    total_train = len(annotations)
    train_image_names = []
    train_image_labels = np.ones((total_train,))

    for index in range(total_train):
        img_name = "train/category_"+str(annotations[index]['category_id']+1)+"/"+str(annotations[index]["image_id"])+".jpg"
        train_image_names.append(img_name)
        train_image_labels[index] = annotations[index]["category_id"]
    train_image_names = np.array(train_image_names)
    # print(test_image_names[:5])
    # print(train_image_names[20:25],":",train_image_labels[:5])
    # release memory
    del annotations
    del test_images
    # train_test_split
    train_x,valid_x,train_y,valid_y = train_test_split(train_image_names,train_image_labels,test_size=0.2, random_state=42)
    
    return train_x,valid_x,train_y,valid_y,test_image_names
# mean and std
def get_dataset_means(train):
#     R_channel_total = 0
#     G_channel_total = 0
#     B_channel_total = 0
#     pixel_count = 0
#     for img_name in train:
#         img = cv2.imread(img_name)
#         if img is not None:
#             img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#             img /= 255.0
#             for row in range(img.shape[0]):
#                 for col in range(img.shape[1]):
#                     pixel_count += 1
#                     R_channel_total += img[row][col][0]
#                     G_channel_total += img[row][col][1]
#                     B_channel_total += img[row][col][2]
    R_mean = 0.436#R_channel_total / pixel_count * 1.0
    G_mean = 0.463#G_channel_total / pixel_count * 1.0
    B_mean = 0.348#B_channel_total / pixel_count * 1.0
    # calculate std
    R_variance = 0.0
    G_variance = 0.0
    B_variance = 0.0
    for img_name in train:
        img = cv2.imread(img_name)
        if img is not None:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img /= 255.0
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    R_variance += pow(img[row][col][0]-R_mean,2)
                    G_variance += pow(img[row][col][1]-G_mean,2)
                    B_variance += pow(img[row][col][2]-B_mean,2)
    #return
    return [R_mean,G_mean,B_mean],[np.sqrt(R_variance),np.sqrt(G_variance),np.sqrt(B_variance)]

def normalize(inputs):
    means,stds = [0.4364117475812634, 0.4637765214112584, 0.3484399638471911],[]
    inputs /= 255.
    inputs -= means
    inputs /= stds
    return inputs

# custom keras generator
class MXGenerator(keras.utils.Sequence):
    def __init__(self,data,n,des_size=(224,224),means=None,stds=None,is_directory=True,batch_size=32,shuffle=True,seed=0):
        '''
        data: tuple of (x,y)
        n: data size
        des_size: standard size
        means: the dataset mean of RGB,default is imagenet means [103.939, 116.779, 123.68]
        batch_size: default is 32
        shuffle: random the data,default is True
        seed: the random seed,default is 0
        '''
        super(MXGenerator,self).__init__()
        self.x = data[0]
        self.y = data[1]
        self.n = n
        self.des_size = des_size
        self.is_directory = is_directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()
        self.means = means
        self.stds = stds

    def reset_index(self):
        self.batch_index = 0
    
    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)
            
    def on_epoch_end(self):
        self._set_index_array()
    
    def __len__(self):
        # batch count
        return self.n // self.batch_size
    # keras will call this function for data if the class is subclass of Sequence,otherwise will call the next
    def __getitem__(self, idx):
        # print("call the getitem function")
        # random choose the index
        if self.seed is not None:
            # affect the np.random.permutation,random generate the index for array
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        # 
        if self.index_array is None:
            self._set_index_array()
        # request the memory
        imgs = np.zeros((self.batch_size,self.des_size[0],self.des_size[1],3))
        # read the data
        if self.is_directory:
            # read from path
            index_array = self.index_array[self.batch_size * idx:self.batch_size * (idx + 1)]
            img_names = self.x[index_array]
            # get the label
            if self.y is not None:
                labels = self.y[index_array]
                labels = keras.utils.to_categorical(labels,997)
            # print(np.array(index_array).shape)
            for name_index in range(len(img_names)): #range(0,((index+1)*self.batch_size - index*self.batch_size))
                img = cv2.imread(img_names[name_index])
                if img is not None:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img,self.des_size)
                    imgs[name_index] = img 
        else:
            # read from path
            index_array = self.index_array[self.batch_size * idx:self.batch_size * (idx + 1)]
            imgs = self.x[index_array]
            labels = keras.utils.to_categorical(self.y[index_array],997)
        if labels is None:
            return imgs
        else:
            return imgs,labels
    # if not use index_generator, this function is not call
    def _flow_index(self):
        # data generate
        while True:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            # batch_index will be set 0 when the value * batch_size large to the n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            
            yield self.index_array[current_index:
                                   current_index + self.batch_size]
    # in python3, __next__,2 is next
    def __next__(self, *args, **kwargs):
        return self.next()
    
    def next(self):
        with self.lock:
            # 
            index_array = next(self.index_generator)
            # self.batch_index += 1
        print("call the next function with next(self.index_generator)")
        return self.__getitem__(self.batch_index)
        
        
