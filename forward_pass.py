import sys
import torch
import math
import config
import argparse
import pickle as pkl
from utils import io
from utils.io import DecodeDataset, eval_coll_fn, SummRating
from torch.utils.data import DataLoader
import os
from os.path import join
from model import hss_seq2seq
from model.hss_classifier import HSSClassifier
from sequence_generator import SequenceGenerator
from tqdm import tqdm
import json
from utils.string_helper import prediction_to_sentence
from utils.io import create_sequence_mask
import nltk
# import rreplace
import torch.nn as nn
import numpy as np
import pickle
from model.hss_model import HSSModel
# from model.multi_task_basic_model import MultiTaskBasicModel
from model.multi_task_basic_classify_seq2seq import MultiTaskBasicClassifySeq2Seq
from model.attn_modulate_classify_seq2seq import AttnModulateClassifySeq2Seq
from model.hre_multi_task_basic_model import HirEncMultiTaskBasicModel
from model.external_feed_classify_seq2seq import ExternalFeedClassifySeq2Seq
from model.external_soft_feed_classify_seq2seq import ExternalSoftFeedClassifySeq2Seq
from model.multi_view_external_soft_feed_classify_seq2seq import MultiViewExternalSoftFeedClassifySeq2Seq
from model.multi_view_attn_modulate_classify_seq2seq import MultiViewAttnModulateClassifySeq2Seq
from model.multi_view_multi_task_basic_seq2seq import MultiViewMultiTaskBasicClassifySeq2Seq
from model.RnnEncSingleClassifier import RnnEncSingleClassifier
from model.seq2seq import Seq2SeqModel
from types import SimpleNamespace
from utils.ordinal_utilities import binary_results_to_rating_preds
from validation import evaluate_loss
from Features import Feature1
from Features import Feature2
from Features import Feature3
from Features import Feature4
from Features import Feature5
from Features import Feature6
from Features import Feature7
from Features import Feature8
from Features import Feature9
from Features import Feature10
UNK_WORD = '<unk>'
import time
def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    # fill time into the name
    if opt.pred_path.find('%s') > 0:
        opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)

    # make directory
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)
        os.makedirs(join(opt.pred_path, 'output'))

    # dump configuration
    torch.save(opt, open(join(opt.pred_path, 'decode.config'), 'wb'))
    json.dump(vars(opt), open(join(opt.pred_path, 'log.json'), 'w'))

    return opt


def init_pretrained_model(pretrained_model_path, opt, rating_tokens_tensor):
    if opt.model_type == 'hss':
        overall_model = HSSModel(opt)
    elif opt.model_type == 'multi_task_basic':
        overall_model = MultiTaskBasicClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "word_attn_modulate":
        overall_model = AttnModulateClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "hre_max":
        overall_model = HirEncMultiTaskBasicModel(opt)
    elif opt.model_type == 'external_feed':
        overall_model = ExternalFeedClassifySeq2Seq(opt)
    elif opt.model_type == "external_soft_feed":
        overall_model = ExternalSoftFeedClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "multi_view_ex_soft_feed":
        overall_model = MultiViewExternalSoftFeedClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "multi_view_attn_modulate":
        overall_model = MultiViewAttnModulateClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "multi_view_multi_task_basic":
        overall_model = MultiViewMultiTaskBasicClassifySeq2Seq(opt, rating_tokens_tensor)
    elif opt.model_type == "rnn_enc_single_classifier":
        overall_model = RnnEncSingleClassifier(opt)
    elif opt.model_type == "seq2seq":
        overall_model = Seq2SeqModel(opt)
    else:
        print(opt.model_type)
        raise ValueError("Invalid model type")
    overall_model.to(opt.device)
    overall_model.load_state_dict(torch.load(pretrained_model_path))
    overall_model.eval()
    return overall_model


def preprocess_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk, src_str_list):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    dec_states = beam_search_result["dec_states"]
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best, dec_state_n_best, oov, src_word_list in zip(predictions, scores, attention, dec_states, oov_lists, src_str_list):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, attn)
            #sentence = [idx2word[int(idx.item())] if int(idx.item()) < vocab_size else oov[int(idx.item())-vocab_size] for idx in pred[:-1]]
            sentences_n_best.append(sentence)
        pred_dict['sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
        pred_dict['attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_dict['dec_states'] = dec_state_n_best # a list of FloatTensor[output sequence length, memory_bank_size], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list


def preprocess_hss_beam_search_result(beam_search_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk, src_str_list):
    batch_size = beam_search_result['batch_size']
    predictions = beam_search_result['predictions']
    scores = beam_search_result['scores']
    attention = beam_search_result['attention']
    sentiment_context = beam_search_result['sentiment_context']  # a list of list, len=(batch, n_best), tensor = [out_seq_len, memory_bank_size], seq_len including eos
    assert len(predictions) == batch_size
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, score_n_best, attn_n_best, senti_n_best, oov, src_word_list in zip(predictions, scores, attention, sentiment_context, oov_lists, src_str_list):
        # attn_n_best: list of tensor with size [trg_len, src_len], len=n_best
        pred_dict = {}
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk, src_word_list, attn)
            #sentence = [idx2word[int(idx.item())] if int(idx.item()) < vocab_size else oov[int(idx.item())-vocab_size] for idx in pred[:-1]]
            sentences_n_best.append(sentence)
        pred_dict['sentences'] = sentences_n_best  # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_dict['scores'] = score_n_best  # a list of zero dim tensor, with len [n_best]
        pred_dict['attention'] = attn_n_best  # a list of FloatTensor[output sequence length, src_len], with len = [n_best]
        pred_dict['sentiment_context'] = senti_n_best  # a list of FloatTensor[output sequence length, memory_bank_size], with len = [n_best]
        pred_list.append(pred_dict)
    return pred_list


def predict(test_data_loader, overall_model, opt):

    merged_rating_preds=np.array([])
    review_rating_preds=np.array([])
    summary_rating_preds=np.array([]) 
    num_exported_samples=0
    path=opt.pretrained_model
    overall_model.to(torch.device('cpu'))
    overall_model.load_state_dict(torch.load(path))
    overall_model.eval()
    with open('weights.pkl','rb') as f:                                                                                                                                                                                             
        W=pickle.load(f)
    with torch.no_grad():
        for batch in tqdm(test_data_loader):
            src = batch['src_tensor']
            src_lens = batch['src_lens']
            src = src.to(opt.device)
            
            if isinstance(overall_model, MultiViewMultiTaskBasicClassifySeq2Seq):
                FeatureVector=np.array([])
                print("len of sentences ", len(batch['src_sent_2d_list'][0]))
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
                FeatureVector=FeatureVector.transpose()

                sentscores=np.dot(FeatureVector, W)
                sumlen=max(math.ceil(0.2*len(batch['src_sent_2d_list'][0])),1)
                sumlen=min(sumlen,3)
                ind = np.argpartition(sentscores, -sumlen)[-sumlen:]
                ind.sort()
                sumlenclassify=max(math.ceil(0.50*len(batch['src_sent_2d_list'][0])),1)  
                GeneratedSummary=""
                
                for i in ind:
                    GeneratedSummary= GeneratedSummary+" "+batch['src_sent_2d_list'][0][i]

                src=[]
                ind = np.argpartition(sentscores, -sumlenclassify)[-sumlenclassify:]
                ind.sort()
                GenSumClassify=""
                for i in ind:
                    GenSumClassify= GenSumClassify+ " "+batch['src_sent_2d_list'][0][i] 
                tokenized= GenSumClassify.strip().split(' ')
                for x in tokenized:
                    src.append(opt.word2idx[x] if x in opt.word2idx else opt.word2idx[UNK_WORD])
                src_lens=[]
                src_lens.append(len(tokenized))
                src2=[]
                src2.append(src)
                src2= np.array(src2)
                src2=torch.LongTensor(src2)
                rt=batch['rating_tensor'][0]
                rating=[]
                rating.append(rt)
                rating=np.array(rating)
                encoder_final_state, classifier_logit, classifier_attention_dist = overall_model(src2, src_lens, None, None, None, None, None, rating, None, None, None)
                summary_classifier_logit = classifier_logit[0] 
                
                summary_class=np.argmax(summary_classifier_logit.detach().cpu().numpy()) if summary_classifier_logit[0] is not None else None

                summary_rating_preds = np.append(summary_rating_preds, summary_class)

  
                with open(join(opt.pred_path, 'output/{}.dec'.format(num_exported_samples)), 'w') as f:
                    f.write(GeneratedSummary)
                num_exported_samples += 1

    if summary_rating_preds is not None:
        with open(join(opt.pred_path, 'summary_rating_output.pkl'), 'wb') as f:
            pkl.dump(summary_rating_preds, f, pkl.HIGHEST_PROTOCOL)


def main(opt):
    # load word2idx and idx2word
    model_dir_path = os.path.dirname(opt.pretrained_model)
    model_dir_path = ''.join(model_dir_path.rsplit('ckpt', 1))
    with open(join(model_dir_path, 'vocab.pkl'), 'rb') as f:
        word2idx = pkl.load(f)
    # load rating_tokens_tensor
    if os.path.exists(os.path.join(model_dir_path, 'rating_tokens_tensor.pt')):
        rating_tokens_tensor = torch.load(os.path.join(model_dir_path, 'rating_tokens_tensor.pt'))
    else:
        rating_tokens_tensor = None

    idx2word = {i: w for w, i in word2idx.items()}
    opt.word2idx = word2idx
    opt.idx2word = idx2word
    opt.vocab_size = len(word2idx)

    # load data
    if not opt.teacher_force_evaluate:
        assert opt.split in ["val", "test"]
        opt.trg_max_len = -1
        opt.src_max_len = -1
        coll_fn_customized = io.summ_rating_flatten_coll_fn(word2idx=word2idx, src_max_len=opt.src_max_len,
                                                            trg_max_len=opt.trg_max_len)
        test_loader = DataLoader(SummRating(opt.split, opt.data), collate_fn=coll_fn_customized, num_workers=opt.batch_workers,
                                  batch_size=1, pin_memory=True, shuffle=False)
    else:
        test_loader = DataLoader(DecodeDataset(opt.split, opt.data), collate_fn=eval_coll_fn(word2idx=word2idx, src_max_len=opt.src_max_len),
                                  num_workers=opt.batch_workers,
                                  batch_size=1, pin_memory=True, shuffle=False)

    # init the pretrained model
    old_opt_dict = json.load(open(join(model_dir_path, "initial.json")))
    old_opt = SimpleNamespace(**old_opt_dict)
    old_opt.word2idx = word2idx
    old_opt.idx2word = idx2word
    old_opt.device = opt.device
    opt.ordinal = old_opt.ordinal
    overall_model = init_pretrained_model(opt.pretrained_model, old_opt, rating_tokens_tensor)

    print("Prediction path: %s" % opt.pred_path)
    if opt.teacher_force_evaluate:
        val_classification_loss_func = nn.NLLLoss(reduction='mean')
        old_opt.inconsistency_loss_type = 'KL_div'
        evaluate_loss(test_loader, overall_model, val_classification_loss_func, old_opt, print_incon_stats=True)
    else:
        predict(test_loader, overall_model, opt)



if __name__ == '__main__':
    # load settings for training
    parser = argparse.ArgumentParser(
        description='predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.predict_opts(parser)
    opt = parser.parse_args()

    opt = process_opt(opt)

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")
    print("start ", time.time())
    start = time.time()
    main(opt)
    end=time.time()
    print("end ", time.time())
    print("TIME TAKEN ", end-start)


