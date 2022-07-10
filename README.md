This is our approach to combine conventional techniques like Evolutionary Algorithms, Rule Based Systems with the contemporary approaches like Deep Learning which require relatively compute intensive training.


We are motivated by the work on those models wherein review summarization and sentiment classification are done in a joint framework and mutually benefit each other in improving the performance. Portion of our source code are adapted from https://github.com/ChenRocks/fast_abs_rl, https://github.com/kenchan0226/dual_view_review_sum .

## Datasets Used

The datasets used can be found at the below URLs
https://www.dropbox.com/s/2b5ueuj6bqpfyvh/min_4_reviews_Sports_and_Outdoors_5.tar.gz?dl=0
https://www.dropbox.com/s/bf1im78iz0cvpta/min_4_Toys_and_Games_5.tar.gz?dl=0
https://www.dropbox.com/s/52wyjfz1xz4oeb4/min_4_Home_and_Kitchen_5.tar.gz?dl=0
https://www.dropbox.com/s/a17hub5m1osdukw/min_4_Movies_and_TV_5.tar.gz?dl=0

If you use the above mentioned datasets, ensure to cite the below two papers:

```
Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering
R. He, J. McAuley
WWW, 2016

Image-based recommendations on styles and substitutes
J. McAuley, C. Targett, J. Shi, A. van den Hengel
SIGIR, 2015
```

## Classifier Training and Genetic Algorithm for optimal weights search

- Compute and export the class distribution on the training set, which will be used by the training script. 
`python3 compute_rating_stat.py -data_dir datasets/processed_reviews_Sports_and_Outdoors_5 -split train`

- Script for training a word2vec embedding on the training set: 
`python3 train_word2vec.py -data datasets/processed_reviews_Sports_and_Outdoors_5 -path word_embeddings/sport_and_outdoors -dim 128`

- There is a two fold training that takes place. Initially the Sentiment Classifier is trained which is later used by Genetic Algotrithm to find the optimal weights for the feature vectors formed corresponding to each sentence in review for summarization task.

Basically there are two modes:[ 'DL', 'GA' ]


Training the sentiment classifier('DL' mode) ==> To run in this mode the following is the argument list 
```
python3 train_ml.py \
-mode=DL\
-data=datasets/processed_reviews_Sports_and_Outdoors_5 \
-exp_path=exp/%s.%s \
-exp=train_movie_dual_view_inc_seed_250 \
-epochs=50 \
-checkpoint_interval=1000 \
-copy_attention \
-batch_size=32 \
-seed=250 \
-w2v=word_embeddings/sport_and_outdoors \
-v_size=50000 \
-word_vec_size=128 \
-encoder_size=256 \
-decoder_size=512 \
-enc_layers=2 \
-residual \
-model_type=multi_view_multi_task_basic \
-dropout=0.0 \
-dec_classify_input_type=dec_state \
-classifier_type=word_multi_hop_attn \
-dec_classifier_type=word_multi_hop_attn \
-gen_loss_weight=0.8 \
-class_loss_weight=0.1 \
-inconsistency_loss_type=KL_div \
-inconsistency_loss_weight=0.1 \
-early_stop_loss=joint \
-batch_workers 0
```

Once the classifier is trained the next step is to run in 'GA' mode to get the optimal feature weights for sentence ranking to for summarization task. The path to the classifier model need to be provided in the argument 'load_from' while running in this mode.

Getting Optimal weights using the Genetic Algorithm('GA' mode) ==> To run in this mode following is the argument list:

```
python3 train_ml.py \
-mode=GA \
-load_from=saved_model/train_movie_dual_view_inc_seed_250.ml.copy.bi-directional.20220315-020304/ckpt/train_movie_dual_view_inc_seed_250.ml.copy.bi-directional-epoch-3-total_batch-13000-joint-0.693 \ 
-data=datasets/processed_reviews_Sports_and_Outdoors_5 \
-exp_path=exp/%s.%s \
-exp=train_movie_dual_view_inc_seed_250 \
-epochs=50 \
-checkpoint_interval=1000 \
-copy_attention \
-batch_size=32 \
-seed=250 \
-w2v=word_embeddings/sport_and_outdoors \
-v_size=50000 \
-word_vec_size=128 \
-encoder_size=256 \
-decoder_size=512 \
-enc_layers=2 \
-residual \
-model_type=multi_view_multi_task_basic \
-dropout=0.0 \
-dec_classify_input_type=dec_state \
-classifier_type=word_multi_hop_attn \
-dec_classifier_type=word_multi_hop_attn \
-gen_loss_weight=0.8 \
-class_loss_weight=0.1 \
-inconsistency_loss_type=KL_div \
-inconsistency_loss_weight=0.1 \
-early_stop_loss=joint \
-batch_workers 0
```


Once the Genetic Algorithm has completed its execution, There will be following files Generated:

```
1. f1_score.pkl - F1 scores for each of the different candidate vector in genetic algorithm
2. balanced_accuracy_score.pkl - Balanced Accuracy scores for each of the different candidate vector in genetic algorithm
3. rouge_scores.pkl - rouge scores obtained when summary was generated using each of the different candidate vector in genetic algorithm as weight vector for sentence score computation.
4. features_vectors.pkl - The actual candidate vector in the Genetic Algorithm
5. total_scores.pkl - The summation of rouge score and F1 score of each of different candidate vector in the genetic algorithm.
```

In our case we chose the corresponding feature vector from 'feature_vectors.pkl' that gave the highest total score in 'total_score.pkl'. Because in our case we wanted those feature vectors that are optimal for both summarization and sentiment classification. 

This feature need to be dumped in file weight.pkl for which dump_weights.py can be used setting the feature vector in the script and running the following command

```
python3 dump_weights.py
```
## Forward Pass ==> Generate Summary and Get Classification Labels

- Download pyrouge, and save it to `path/to/pyrouge`. 
`git clone https://github.com/andersjo/pyrouge.git`

- Export ROUGE score enviornment variable
`export ROUGE=[path/to/pyrouge/tools/ROUGE-1.5.5]`

- Make evaluation reference for a dataset (Only need to do it for once for each dataset)
`python make_eval_reference.py -data datasets/processed_reviews_Sports_and_Outdoors_5 -split all`

- Run predict, specify the path to the best checkpoint (lowest validation loss) in the `-pretrained_model` argument. 
```
python forward_pass.py \
-data datasets/processed_reviews_Sports_and_Outdoors_5 \
-pred_path pred/%s.%s \
-exp predict_dual_view_inc_seed_250 \
-pretrained_model saved_model/train_movie_dual_view_inc_seed_250.ml.copy.bi-directional.20220315-020304/ckpt/train_movie_dual_view_inc_seed_250.ml.copy.bi-directional-epoch-3-total_batch-13000-joint-0.693 \
-seed 9527 \
-batch_size 1 \
-replace_unk \
-src_max_len -1
```

- Run evaluate prediction to compute ROGUE scores, macro F1, and balanced accuracy. 
`python evaluate_prediction.py -rouge -decode_dir pred/predict_dual_view_inc_seed_250.20190901-160022 -data datasets/processed_reviews_Sports_and_Outdoors_5`

