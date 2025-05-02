import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import os
import json
import pandas as pd
import ijson 
import shutil 

USER_DATA_PATH = '../datasets/Twibot-22/user.json'
ID_TWEET_PATH = './processed_data/id_tweet.json' 
DES_TENSOR_PATH = './processed_data/des_tensor.pt' 
TWEETS_TENSOR_PATH = './processed_data/tweets_tensor.pt' 

USER_EMBEDDINGS_TEMP_DIR = './temp_user_embeddings'

user = pd.read_json(USER_DATA_PATH)
user_text = list(user['description'])

feature_extract = pipeline('feature-extraction', model='roberta-base', tokenizer='roberta-base', device=1, padding=True, truncation=True, max_length=50, add_special_tokens=True)


def Des_embbeding():
        print('Running feature1 embedding')
        path="./processed_data/des_tensor.pt"
        if not os.path.exists(path):
            des_vec=[]
            for k,each in enumerate(tqdm(user_text)):
                if each is None:
                    des_vec.append(torch.zeros(768))
                else:
                    feature=torch.Tensor(feature_extract(each))
                    for (i,tensor) in enumerate(feature[0]):
                        if i==0:
                            feature_tensor=tensor
                        else:
                            feature_tensor+=tensor
                    feature_tensor/=feature.shape[1]
                    des_vec.append(feature_tensor)
                    
            des_tensor=torch.stack(des_vec,0)
            torch.save(des_tensor,path)
        else:
            des_tensor=torch.load(path)
        print('Finished')
        return des_tensor


def tweets_embedding():
    print('Running feature2 embedding')
    path = "./processed_data/tweets_tensor.pt"

    if not os.path.exists(path):
        if not os.path.exists(USER_EMBEDDINGS_TEMP_DIR):
            os.makedirs(USER_EMBEDDINGS_TEMP_DIR)

        with open(ID_TWEET_PATH, 'rb') as f:
            user_tweets_iterator = ijson.kvitems(f, '')
            for user_index_str, each_user_tweets_list in tqdm(user_tweets_iterator):
                total_each_person_tweets = torch.zeros(768)
                valid_tweet_count = 0

                if each_user_tweets_list:
                    for j, each_tweet in enumerate(each_user_tweets_list):
                        if j >= 20:
                            break
                        if each_tweet is None:
                            continue

                        tweet_embedding_output = feature_extract(each_tweet)
                        if tweet_embedding_output and len(tweet_embedding_output) > 0 and len(tweet_embedding_output[0]) > 0:
                            feature = torch.tensor(tweet_embedding_output[0])
                            total_word_tensor = torch.sum(feature, dim=0) / feature.shape[0]
                            total_each_person_tweets += total_word_tensor
                            valid_tweet_count += 1

                if valid_tweet_count > 0:
                    total_each_person_tweets /= valid_tweet_count

                user_index = int(user_index_str)
                temp_embedding_file_path = os.path.join(USER_EMBEDDINGS_TEMP_DIR, f'user_{user_index}.pt')
                torch.save(total_each_person_tweets, temp_embedding_file_path)

        consolidated_embeddings_list = []
        saved_embedding_files = [f for f in os.listdir(USER_EMBEDDINGS_TEMP_DIR) if f.startswith('user_') and f.endswith('.pt')]
        saved_embedding_files.sort(key=lambda f: int(f.replace('user_', '').replace('.pt', '')))
        for file_name in tqdm(saved_embedding_files):
            file_path = os.path.join(USER_EMBEDDINGS_TEMP_DIR, file_name)
            user_embedding = torch.load(file_path, map_location=device)
            consolidated_embeddings_list.append(user_embedding)

        tweet_tensor = torch.stack(consolidated_embeddings_list, dim=0)
        torch.save(tweet_tensor, path)

    else:
        tweet_tensor = torch.load(path, map_location=device)

    print('Finished')
    return tweet_tensor

des_tensor = Des_embbeding()
tweets_tensor = tweets_embedding()

print(f"Description tensor shape: {des_tensor.shape}")
print(f"Tweets tensor shape: {tweets_tensor.shape}")
