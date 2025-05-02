Model two is BotRGCN, a Twitter bot detection[1].
This model is obtained from the twitbo22 benchmark[2].

Requirements

 ```pip: pip install -r requirements.txt ```

 
 ```conda : conda install --yes --file requirements.txt ```

How to reproduce:

specify the dataset by entering corresponding folder

 ```cresci-15 : cd cresci_15/```

 
 ```twibot-22 : cd twibot_22/```

preprocess the dataset by running:

 ```python preprocess.py ```

train BotRGCN model by running:

 ```python train.py ```


Notice:

1. For running cresci2015, it is better to use instruction by [3].
3. For running twibot_22, if your device doesn't have high memory size (64GB +), it is better to use ijson to handle json files (dataset has 10 x 10GB json files that need to merge).
4. changes for handling memory usuage is applied and you can see them in preprocessing_111_final, preprocessing_2222_final, and train_4_final.

Refrences:

1.Feng, S., Wan, H., Wang, N., & Luo, M. (2021, November). BotRGCN: Twitter bot detection with relational graph convolutional networks. In Proceedings of the 2021 IEEE/ACM international conference on advances in social networks analysis and mining (pp. 236-239).

2.https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master

3.https://github.com/travistangvh/TwitterBotBusters/tree/master

