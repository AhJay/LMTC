import torch
from torch.utils.data import DataLoader
from dataloader import *
from model import LSTMClassifier
from vocabulary import *
from textCNN import *
from fast import *
from utils import *
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

batch = 2
torch.backends.cudnn.enabled = False
#model = textCNN(100,60669,9807,batch)
model = textCNN(100,35560,4271,batch)
#model = fastText(100,100,35560,4271,batch)
#model.load_state_dict(torch.load('text_cnn_word.pkl'))
#model = LSTMClassifier(100,100,35560,4271,batch)
#model.load_state_dict(torch.load('bi_att.pkl'))
model = model.to(device)


tot_epoch =30
criterion = torch.nn.BCEWithLogitsLoss()
#criterion = standard_loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0015)
scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
train_data = MyDataset('train.npy',flg=1)
#train_data = MyDataset('../data/Eurlex/train_drop_small.npy',flg=1,tot=tot_epoch-1)

test_data = MyDataset('test.npy',0)
val_data = MyDataset('dev.npy',flg=1)

train_loader = DataLoader(train_data, batch_size=batch,num_workers=1,shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch,num_workers=1,shuffle=False)
val_loader = DataLoader(val_data,  batch_size=batch,num_workers=1,shuffle=False)
prev_tot = float('inf')
epoch = 0

prev = 0

import time
beg = time.time()
while epoch < tot_epoch:
    model.train()
    epoch += 1
    tot_loss2 = 0
    tot_loss = 0
    tot_ndcg = 0
    tot_rp=0
    cnt=0


    for idx, (text,target,target2) in enumerate(train_loader):
        
        text = text.long().to(device)
        target = target.to(device)
        target2 = target2.to(device) 
       
        cnt += 1
        score = model.forward(text,target)
    
        loss = criterion(score,target)  + (1-epoch/tot_epoch) * criterion(score,target2)
    

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #tot_loss2 += float(loss2)
        tot_loss += float(loss)
    
        if epoch%5 ==0:
            tot_rp +=  mean_rprecision_k(target.cpu().detach().numpy(),score.cpu().detach().numpy())
            tot_ndcg += mean_ndcg_score(target.cpu().detach().numpy(),score.cpu().detach().numpy())
        torch.cuda.empty_cache()

    scheduler.step()
    
    if epoch%5 == 0:
        print(epoch,tot_loss/cnt, tot_ndcg/cnt, tot_rp/cnt)
    else:
        print(epoch,tot_loss/cnt,tot_loss2/cnt)


    with torch.no_grad():
        model.eval()

        if epoch % 5 == 0 or epoch==9:
            tot_loss = 0
            cnt = 0
            tot_ndcg = 0
            tot_rp = 0
            for idx, (text,target,target2) in enumerate(val_loader):
                cnt += 1
                text = text.long().to(device)
                target = target.to(device)

                score= model.forward(text,target)

                tot_loss += float(loss)
                tot_ndcg += mean_ndcg_score(target.cpu().detach().numpy(),score.cpu().detach().numpy())
                tot_rp +=  mean_rprecision_k(target.cpu().detach().numpy(),score.cpu().detach().numpy())


                torch.cuda.empty_cache()
            print('validation', tot_ndcg/cnt, tot_rp/cnt)
            if tot_ndcg/len(val_loader) > prev:
                print('saved')
                torch.save(model.state_dict(), 'text_cnn.pkl')
                prev = tot_ndcg/len(val_loader)
