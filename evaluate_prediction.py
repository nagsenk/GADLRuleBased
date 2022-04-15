""" Evaluate the baselines ont ROUGE/METEOR"""
""" Adapted from https://github.com/ChenRocks/fast_abs_rl """
import argparse
import json
import os
from os.path import join, exists
import numpy as np
import pickle as pkl
from utils.evaluate import eval_meteor, eval_rouge
from utils.utils_glue import acc_and_micro_f1, acc_and_macro_f1, balanced_acc_and_macro_f1


def compute_rating_metric(decode_dir, data_dir, split):
    # open ground-truth rating file
    ground_truth_rating_dir = join(data_dir, 'ratings', split)
    with open(join(ground_truth_rating_dir, 'gold_ratings.pkl'), 'rb') as f:
        out_label_ids = pkl.load(f)
    dec_rating_file = join(decode_dir, 'summary_rating_output.pkl')
    if exists(dec_rating_file):
        with open(dec_rating_file, 'rb') as f:
            dec_rating_preds = pkl.load(f)
        # compute F1 and acc
        dec_classification_result = acc_and_macro_f1(dec_rating_preds, out_label_ids)
        balanced_dec_cf_result = balanced_acc_and_macro_f1(dec_rating_preds, out_label_ids)
        dec_macro_f1 = dec_classification_result['f1']
        dec_acc = dec_classification_result['acc']
        balanced_dec_acc = balanced_dec_cf_result['acc']
        dec_rating_results_output = "summary-view macro f1 score:\t{:.4f}\ndec accuracy:\t{:.4f}\nbalanced_dec_acc:\t{:.4f}\n".format(dec_macro_f1, dec_acc, balanced_dec_acc)
        print(dec_rating_results_output)
        with open(join(args.decode_dir, 'dec_rating_results.txt'), 'w') as f:
            f.write(dec_rating_results_output)
    else:
        print("{} does NOT exist!".format(dec_rating_file))

def main(args):
    dec_dir = join(args.decode_dir, 'output')
    with open(join(args.decode_dir, 'log.json')) as f:
        split = json.loads(f.read())['split']
    ref_dir = join(args.data, 'refs', split)
    assert exists(ref_dir)

    if args.rouge:
        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'rouge'
        print(output)
        with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
            f.write(output)
    elif args.meteor:
        dec_pattern = '[0-9]+.dec'
        ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'meteor'
        print(output)
        with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
            f.write(output)

    compute_rating_metric(args.decode_dir, args.data, split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the output files for the RL full models')

    parser.add_argument('-rouge', action='store_true',
                            help='ROUGE evaluation')
    parser.add_argument('-meteor', action='store_true',
                            help='METEOR evaluation')
    parser.add_argument('-decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('-data', action='store', required=True,
                        help='directory of decoded summaries')

    args = parser.parse_args()
    main(args)
