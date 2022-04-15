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
from utils.report import export_train_and_valid_loss
from utils.io import remove_old_ckpts
from utils.inconsistency_loss import inconsistency_loss_func

EPS = 1e-8


def train_model(overall_model, optimizer_ml, train_data_loader, valid_data_loader, opt, train_classification_loss_func, val_classification_loss_func):
    exp = opt.exp.split('.')[0]

    # make the code compatible when tensorboardX is not available
    try:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(comment='_' + exp)
    except ModuleNotFoundError:
        tb_writer = None

    logging.info('======================  Start Training  =========================')

    total_batch = -1
    early_stop_flag = False

    report_train_loss_statistics = JointLossStatistics()

    best_valid_joint_loss = float('inf')
    num_stop_dropping = 0

    if opt.train_from:  # opt.train_from:
        #TODO: load the training state
        raise ValueError("Not implemented the function of load from trained model")
        pass

    overall_model.train()

    final_best_valid_joint_loss = float('inf')
    final_correspond_valid_ppl = float('inf')
    final_correspond_enc_class_loss = 0.0
    final_correspond_enc_class_f1 = 0.0
    final_correspond_dec_class_loss = 0.0
    final_correspond_dec_class_f1 = 0.0
    final_inconsist_loss = 0.0

    previous_valid_joint_loss = float('inf')
    previous_valid_ppl = float('inf')
    previous_valid_enc_f1 = 0.0

    for epoch in range(opt.start_epoch, opt.epochs+1):
        if early_stop_flag:
            break
        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1
            batch_loss_stat= train_one_batch(batch, overall_model, optimizer_ml, opt, total_batch, train_classification_loss_func, tb_writer)
            report_train_loss_statistics.update(batch_loss_stat)
    
            if total_batch % opt.checkpoint_interval == 0:
                logging.info("Epoch %d; batch: %d; total batch: %d" % (epoch, batch_i, total_batch))
                sys.stdout.flush()
            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):
                    logging.info("Enter check point!")
                    sys.stdout.flush()
                    train_enc_class_loss = report_train_loss_statistics.enc_classification_loss()
                    train_joint_loss = report_train_loss_statistics.joint_loss()
                    valid_loss_stat, valid_class_result = evaluate_loss(valid_data_loader, overall_model, val_classification_loss_func, opt)
                    overall_model.train()
                    valid_enc_class_loss = valid_loss_stat.enc_classification_loss()
                    valid_joint_loss = valid_loss_stat.joint_loss()

                    valid_enc_f1 = valid_class_result['acc']
                    valid_f1 = valid_enc_f1
                   
                    if math.isnan(valid_joint_loss) or math.isnan(train_joint_loss):
                        logging.info(
                            "NaN valid loss. Epoch: %d; batch_i: %d, total_batch: %d" % (epoch, batch_i, total_batch))
                        exit()
                    # print out train and valid loss
                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        'training enc class loss: %.3f; valid enc class loss: %.3f; valid enc class f1: %.3f' % (
                            train_enc_class_loss, valid_enc_class_loss, valid_enc_f1))
                   
                    if opt.early_stop_loss == "joint":
                        previous_valid_loss = previous_valid_joint_loss
                        current_valid_loss = valid_joint_loss
                    elif opt.early_stop_loss == "ppl":
                        previous_valid_loss = previous_valid_ppl
                        current_valid_loss = valid_ppl
                    else:
                        raise ValueError

                    if epoch >= opt.start_decay_and_early_stop_at:

                        if current_valid_loss < previous_valid_loss: # update the best valid loss and save the model parameters

                            logging.info("Valid loss drops")
                            sys.stdout.flush()
                            num_stop_dropping = 0
                            check_pt_model_path = os.path.join(opt.model_path, 'ckpt', '%s-epoch-%d-total_batch-%d-%s-%.3f' % (
                                opt.exp, epoch, total_batch, opt.early_stop_loss, current_valid_loss))
                            torch.save(  # save model parameters
                                overall_model.state_dict(),
                                open(check_pt_model_path, 'wb')
                            )
                            logging.info('Saving checkpoint to %s' % check_pt_model_path)
                        else:
                            logging.info("Valid loss does not drop")
                            sys.stdout.flush()
                            num_stop_dropping += 1
                            # decay the learning rate by a factor
                            if opt.learning_rate_decay < 1:
                                for i, param_group in enumerate(optimizer_ml.param_groups):
                                    old_lr = float(param_group['lr'])
                                    new_lr = old_lr * opt.learning_rate_decay
                                    if new_lr < opt.min_lr:
                                        new_lr = opt.min_lr
                                    if old_lr - new_lr > EPS:
                                        param_group['lr'] = new_lr
                                logging.info('Learning rate drops to {}'.format(new_lr))

                        previous_valid_joint_loss = valid_joint_loss
                        previous_valid_enc_f1 = valid_enc_f1


                        best_condition1 = opt.early_stop_loss == 'joint' and valid_joint_loss < final_best_valid_joint_loss
                        if best_condition1 :
                            best_valid_joint_loss = valid_joint_loss
                            final_best_valid_joint_loss = valid_joint_loss
                            final_correspond_enc_class_loss = valid_enc_class_loss
                            final_correspond_enc_class_f1 = valid_enc_f1
             

                        if not opt.disable_early_stop:
                            if num_stop_dropping >= opt.early_stop_tolerance:
                                logging.info('Have not increased for %d check points, early stop training' % num_stop_dropping)
                                early_stop_flag = True
                                break

                    report_train_loss_statistics.clear()
    logging.info("final_correspond_enc_class_loss: %.3f" % final_correspond_enc_class_loss)
    logging.info("final_correspond_enc_class_balancy: %.3f" % final_correspond_enc_class_f1)
    remove_old_ckpts(opt.model_path, reverse=False)


def train_one_batch(batch, overall_model, optimizer, opt, global_step, classification_loss_func, tb_writer):
    # src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_sent_2d_list, trg, trg_oov, trg_lens, trg_mask, rating, _ = batch

    # changed by wchen to a dictionary batch
    src = batch['src_tensor']
    src_lens = batch['src_lens']
    src_mask = batch['src_mask']
    src_sent_positions = batch['src_sent_positions']
    src_sent_nums = batch['src_sent_nums']
    src_sent_mask = batch['src_sent_mask']
    src_oov = batch['src_oov_tensor']
    oov_lists = batch['oov_lists']
    src_str_list = batch['src_list_tokenized']
    trg_sent_2d_list = batch['tgt_sent_2d_list']
    trg = batch['tgt_tensor']
    trg_oov = batch['tgt_oov_tensor']
    trg_lens = batch['tgt_lens']
    trg_mask = batch['tgt_mask']
    rating = batch['rating_tensor']
    indices = batch['original_indices']

    """
    trg: LongTensor [batch, trg_seq_len], each target trg[i] contains the indices of a set of concatenated keyphrases, separated by opt.word2idx[io.SEP_WORD]
                 if opt.delimiter_type = 0, SEP_WORD=<sep>, if opt.delimiter_type = 1, SEP_WORD=<eos>
    trg_oov: same as trg_oov, but all unk words are replaced with temporary idx, e.g. 50000, 50001 etc.
    """
    #seq2seq_model = overall_model['generator']
    #classifier_model = overall_model['classifier']
    batch_size = src.size(0)
    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)

    src_sent_positions = src_sent_positions.to(opt.device)
    src_sent_mask = src_sent_mask.to(opt.device)

    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)
    rating = rating.to(opt.device)

    optimizer.zero_grad()

    start_time = time.time()

    # forward
    if overall_model.model_type == 'hre_max':
         encoder_final_state,  classifier_logit, classifier_attention_dist = \
            overall_model(src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, src_sent_positions, src_sent_nums, src_sent_mask)
    else:
         encoder_final_state, classifier_logit, classifier_attention_dist = \
            overall_model(src, src_lens, trg, src_oov, max_num_oov, src_mask, trg_mask, rating, src_sent_positions, src_sent_nums, src_sent_mask)

    forward_time = time_since(start_time)

    start_time = time.time()

    # compute loss of classification
    if classifier_logit is not None:
        if isinstance(classifier_logit, tuple):
            enc_normalized_classification_loss = classification_loss_func(classifier_logit[0], rating)  # normalized by batch size already

        else:
            enc_normalized_classification_loss = classification_loss_func(classifier_logit, rating)  # normalized by batch size already

    else:
        enc_normalized_classification_loss = torch.Tensor([0.0]).to(opt.device)

    loss_compute_time = time_since(start_time)

    start_time = time.time()
    # back propagation on the joint loss
    enc_normalized_classification_loss.backward()

    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(overall_model.parameters(), opt.max_grad_norm)

    optimizer.step()

    # log each loss to tensorboard
    if tb_writer is not None:
        tb_writer.add_scalar('enc_classification_loss', enc_normalized_classification_loss.item(), global_step)

    # construct a statistic object for the loss
    stat = JointLossStatistics(enc_normalized_classification_loss.item(),0.0, enc_normalized_classification_loss.item(), 0.0, 0.0,
                               n_iterations=1, n_tokens=0, forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time)

    # decoder_dist_out = decoder_dist.detach() if decoder_dist is not None else None
    return stat
