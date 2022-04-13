import torch.nn as nn
from utils.masked_loss import masked_cross_entropy
from utils.statistics import JointLossStatistics
from utils.time_log import time_since
from validation import evaluate_loss
import time
import math
import logging
import torch
import sys
import os
import metric
import numpy as np
from utils.report import export_train_and_valid_loss
from utils.io import remove_old_ckpts
from utils.inconsistency_loss import inconsistency_loss_func
from geneticalgorithm import geneticalgorithm as ga
from Features import Feature1
from Features import Feature2
from Features import Feature3
from Features import Feature4
from Features import Feature7
from Features import Feature8
from Features import Feature6
from Features import Feature5
from Features import Feature9
from Features import Feature10
import nltk
import pickle
from utils.ordinal_loss import OrdinalMSELoss, OrdinalXELoss
EPS = 1e-8
from rouge_score import rouge_scorer
import operator
sloss=[]
closs=[]
optimizer_ml=None 
global train_data_loader
valid_data_loader=None 
opt=None 
train_classification_loss_func=None 
val_classification_loss_func=None
batch=None
feats=[]
tloss=[]
F1=[]
from utils.utils_glue import acc_and_micro_f1, acc_and_macro_f1, balanced_acc_and_macro_f1
Balanced_Accuracy=[]
UNK_WORD = '<unk>'

def F(X):
  
    summary_generation_loss_train=0.0
    classification_loss_train=0.0
    X=X.transpose()
    UNK_WORD = '<unk>'
    feats.append(X)
    rouge_total=0.0
    class_total=0.0
    y_pred=[]
    y_true=[]

    print("Weight Vector:")
    print(X)    
    with open('tdl.pkl', 'rb') as f:
        train_data_loader=pickle.load(f)
    with open('vdl.pkl', 'rb') as f: 

        valid_data_loader=pickle.load(f)
    with open('ValidFeat.pkl', 'rb') as f: 
        ValidFeat=pickle.load(f)
    with open('word2idx.pkl', 'rb') as f:
        word2idx=pickle.load(f)
    batch_i=0
    batch_i=int(batch_i)
    with open('model.pkl', 'rb') as f:                                                                                                                                                                                                                   overall_model=pickle.load(f)
    overall_model.to(torch.device('cpu'))
    overall_model.load_state_dict(torch.load(opt.load_from))
    overall_model.eval()
    for batch in valid_data_loader:
        if batch_i >1000:
               break
        fetvsent=ValidFeat[batch_i]                                                                                                                                                                                                   
        sentscores=np.dot(fetvsent, X)
      
        sumlen=max(math.ceil(0.2*len(batch['src_sent_2d_list'][0])),1)
        ind = np.argpartition(sentscores, -sumlen)[-sumlen:]
        ind.sort()
        GeneratedSummary=""
        for i in ind:
            GeneratedSummary= GeneratedSummary+" "+batch['src_sent_2d_list'][0][i]
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
        scores = scorer.score(batch['tgt_sent_2d_list'][0][0],GeneratedSummary)
        rouge_total+=scores['rouge1'][1]
        
        expected=np.array([])
        for sent in batch['src_sent_2d_list'][0]:
            expected=np.append(expected,scorer.score(batch['tgt_sent_2d_list'][0][0],sent)['rouge1'][1])
        #normalize expected and sentscores
        rt=batch['rating_tensor'][0]
        if np.max(expected) > 0.0:
                norm = np.max(expected)
                expected = expected/norm
        if np.max(sentscores) > 0.0:
                norm = np.max(sentscores)
                sentscores =  sentscores/norm
        
        tokenized= GeneratedSummary.strip().split(' ')
        src=[]
        for x in tokenized:
                src.append(word2idx[x] if x in word2idx else word2idx[UNK_WORD]) 
       
        src_lens=[]      
        src_lens.append(len(tokenized))
        src2=[]
        src2.append(src)
        src2= np.array(src2)
        src2=torch.LongTensor(src2)
     
        rating=[] 
        rating.append(rt)
        rating=np.array(rating)
        #Summary is generated now we will pass through encoder and then through classifier
        encoder_final_state, classifier_logit, classifier_attention_dist = overall_model(
                src2, src_lens, None, None, None, None, None, rating, None, None, None)
        enc_classifier_logit = classifier_logit[0]
        
        classification_loss_func=OrdinalMSELoss(5,torch.device("cpu"))
       
        
        l1=classifier_logit[0]
        l2=[]
        l2.append(batch['rating_tensor'][0])
        l2=torch.LongTensor(l2)
        l1=torch.Tensor(l1)
  
        y_true.append(batch['rating_tensor'][0])     

        obtained=np.argmax(classifier_logit[0].detach().cpu().numpy()) if classifier_logit[0] is not None else None
        y_pred.append(obtained)
        enc_normalized_classification_loss = classification_loss_func(l1, l2)
        if obtained== batch['rating_tensor'][0]:
                class_total+=1
     
        summary_generation_loss_train+=((np.linalg.norm(expected-sentscores)**2)/len(expected))*100
        classification_loss_train+=enc_normalized_classification_loss
        batch_i+=1
      
    total_samples=batch_i
  
    D= balanced_acc_and_macro_f1(y_pred,y_true)
    F1.append(D['f1'])
    Balanced_Accuracy.append(D['acc'])

    with open('F1.pkl','wb') as f:
        pickle.dump(F1,f)
    with open('BA.pkl','wb') as f:
        pickle.dump(Balanced_Accuracy,f)    
   
    closs.append(D['acc_and_f1'])
    sloss.append(rouge_total/total_samples)
    tloss.append(D['acc_and_f1']+rouge_total/total_samples)
    with open('sloss.pkl','wb') as f:
        pickle.dump(sloss,f)
    with open('closs.pkl','wb') as f:
        pickle.dump(closs,f)
    with open('feats.pkl','wb') as f:
        pickle.dump(feats,f)
    with open('tloss.pkl','wb') as f:
        pickle.dump(tloss,f)
    
    return -(rouge_total/total_samples+ D['acc_and_f1'])

 
def FeatureExtraction(data_loader, tov):

    TrainFeat=[]
    ValidFeat=[]
    c=0
    tdl=[]
    vdl=[]

    for batch_i, batch in enumerate(data_loader):
     
        FeatureVector=np.array([])
        c+=1
        f1=Feature1(batch['src_sent_2d_list'][0])
        FeaturVector=f1

        f2=Feature2(batch['src_sent_2d_list'][0])
        FeatureVector=np.vstack((f1,f2))

        f3=Feature3(batch['src_sent_2d_list'][0])
        FeatureVector=np.vstack((FeatureVector,f3))        

        f4=Feature4(batch['src_sent_2d_list'][0])
        FeatureVector=np.vstack((FeatureVector,f4))
       
        f5=Feature5(batch['src_sent_2d_list'][0])
        FeatureVector=np.vstack((FeatureVector,f5))

        f6=Feature6(batch['src_sent_2d_list'][0])
        FeatureVector=np.vstack((FeatureVector,f6))
 
        f7=Feature7(batch['src_sent_2d_list'][0])
        FeatureVector=np.vstack((FeatureVector,f7))
               
        f8=Feature8(batch['src_sent_2d_list'][0])
        FeatureVector=np.vstack((FeatureVector,f8))

        f9=Feature9(batch['src_sent_2d_list'][0])
        FeatureVector=np.vstack((FeatureVector,f9))

        f10=Feature10(batch['src_sent_2d_list'][0])
        FeatureVector=np.vstack((FeatureVector,f10))
        if(tov=="train"):
                tdl.append(batch)
        else:
                vdl.append(batch)
        FeatureVector=FeatureVector.transpose()
        if tov=="train":
            TrainFeat.append(FeatureVector)
        else:
            ValidFeat.append(FeatureVector)

    if tov=="train":
                print(TrainFeat)
                with open('TrainFeat.pkl', 'wb') as f:
                                pickle.dump(TrainFeat, f)
                with open('tdl.pkl', 'wb') as f: 
                                pickle.dump(tdl,f)
    else:
                print(ValidFeat)
                with open('ValidFeat.pkl','wb') as f:
                                pickle.dump(ValidFeat,f)
                with open('vdl.pkl','wb') as f:
                                pickle.dump(vdl,f)

def GeneticAlgorithm(overall_model, optimizer_ml, train_data_loader, valid_data_loader, opt_inc, train_classification_loss_func, val_classification_loss_func):
    varbound=np.array([[0,1]]*10)
    overall_model=overall_model
    optimizer_ml=optimizer_ml
    with open('train_data_loader.pkl', 'wb') as f:
        pickle.dump(train_data_loader,f)
    with open('valid_data_loader.pkl','wb') as f:
        pickle.dump(valid_data_loader,f)
    global opt
    opt=opt_inc
    train_classification_loss_func=train_classification_loss_func
    val_classification_loss_func=val_classification_loss_func
    with open('train_data_loader.pkl', 'rb') as f:                                                                                                                                                               
        train_data_loader=pickle.load(f)
    with open('valid_data_loader.pkl','rb') as f:
        valid_data_loader=pickle.load(f)
    with open('model.pkl','wb') as f:
        pickle.dump(overall_model, f)
   # FeatureExtraction(train_data_loader,"train")
    print("### <<< ----    [ Performing Feature Extraction on Validation Data ]    ---- >>> ###")
    FeatureExtraction(valid_data_loader, "valid")
    print(" Genetic Algorithm in Progress ")
    model=ga(function=F,dimension=10,variable_type='real',function_timeout=10000, variable_boundaries=varbound)
    # model=ga(function=f,dimension=30,variable_type='bool')
    model.run()
