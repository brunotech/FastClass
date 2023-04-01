import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from sentence_transformers import SentenceTransformer, util
import csv
from collections import Counter
import pickle
import time
import faiss
import numpy as np
import pandas as pd
import argparse
import heapq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='sst')
    parser.add_argument("--path", type=str, default='your_path/FastClass/data/sst')
    args = parser.parse_args()
    print('------------------------------------------Task external data select---------------------------------------------')
    print(vars(args))
    for top_k_hits in [100, 200, 300]:
        news_data, news_score = {}, {}
        with open(os.path.join(args.path, f'train_data_{top_k_hits}.txt'), 'r', encoding='utf-8') as f:
            for lines in f:
                if int(lines.strip().split('\t')[0]) in news_data:
                    news_data[int(lines.strip().split('\t')[0])].append(lines.strip().split('\t')[2])
                    news_score[int(lines.strip().split('\t')[0])].append(lines.strip().split('\t')[1])


                else:
                    news_data[int(lines.strip().split('\t')[0])] = [lines.strip().split('\t')[2]]
                    news_score[int(lines.strip().split('\t')[0])] = [lines.strip().split('\t')[1]]
        final_300 = pd.DataFrame(columns=['target', 'score', 'data'])
        final_500 = pd.DataFrame(columns=['target', 'score', 'data'])
        final_800 = pd.DataFrame(columns=['target', 'score', 'data'])
        for k, data in news_data.items():
            sim_index = list(map(news_score[k].index, heapq.nlargest(800, news_score[k])))

            final_data, final_score = [], []
            for idx in sim_index:
                final_data.append(data[idx])
                final_score.append(news_score[k][idx])

            final_300['target'] = [k]*300
            final_300['score'] = final_score[:300]
            final_300['data'] = final_data[:300]

            final_500['target'] = [k] * 500
            final_500['score'] = final_score[:500]
            final_500['data'] = final_data[:500]

            final_800['target'] = [k] * 800
            final_800['score'] = final_score[:800]
            final_800['data'] = final_data[:800]

        final_300.to_csv(f'mix_{top_k_hits}_300.tsv', sep='\t', index=False)
        final_500.to_csv(f'mix_{top_k_hits}_500.tsv', sep='\t', index=False)
        final_800.to_csv(f'mix_{top_k_hits}_800.tsv', sep='\t', index=False)


