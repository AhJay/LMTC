import torch
from collections import defaultdict
import torch.utils.data as data
import os
import pickle
import nltk
import numpy as np
import random
import pickle




class MyDataset(data.Dataset):
  
    def __init__(self,path, flg):

        self.f = np.load('train.npy', allow_pickle=True)

        self.w = np.load('./label_coef_word.npy',allow_pickle=True).item()


        self.vocab = pickle.load(open('vocab.pkl','rb'))
    
        self.flg = flg
        
        

        

    def __getitem__(self, index):
        
        text = self.f[index][0]
        target = self.f[index][1:]
        length= 4271
        labels = torch.zeros([length])
        labels2 = torch.zeros([length])
    
       
        tokens = nltk.tokenize.word_tokenize(text.lower())
        text = []
        text.append(self.vocab('<start>'))
        text.extend([self.vocab(token) for token in tokens])
        if len(text) > 500:
            text = text[:500]
        else:
            while len(text) < 500:
                 text.append(self.vocab('<pad>'))

        text.append(self.vocab('<end>'))
        text = torch.Tensor(text)
        
         
        if self.flg:
            return text,labels,labels2
        else:
            return text,labels,labels

    def __len__(self):
        return len(self.f)




def get_loader(path,batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    data = MyDataset(path)
    
 
    data_loader = torch.utils.data.DataLoader(dataset=data, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader
