<h1 align="center">Social Bot Detection Project</h1>

Twitter has long struggled with the issue of automated users (bots). These bots can artificially amplify trends, spread misinformation, and are often used by bot farms to push political agendas (ex, state-sponsored influence operations like those attributed to Russia). As bots evolve, earlier detection models often become ineffective.

Detecting these bots is critical. Agencies like the FBI and other government organizations invest resources into uncovering these bot networks. Our goal is to apply domain adaptation to improve the performance of existing Twitter bot detection models, and then compare our updated model’s results to previous benchmarks.

Midproject presentation - https://youtu.be/nlXt7jhy3c8

Final presentation - 

## How to use our Code
Model One 2015 & 2023 - Download all files in the folder and open the .ipynb file in either Google Colab or Jupyter Notebook. Insert the CSV datasets into the environment, and run all code blocks sequentially.

...

## Model One Architecture
Input Layer: 10 features
Hidden Layers: 500 → 200 neurons (Dense)
Output Layer: 1 neuron (sigmoid)
Total Parameters: 105,901

## 2015 Dataset
Applied domain adaptation using a 2015 dataset with labeled fake accounts and real accounts.
Uses the following selected features: {age, location, is verified, total tweets, total following, total followers, total likes, has avatar, has background, is protected}
Combined test set from both datasets
Uses binary cross entropy loss and adam

## 2023 Dataset
Applied domain adaptation using a 2023 dataset with labeled fake accounts and real accounts.
Uses the following selected features: {"avg_tweet_length", "avg_retweets", "avg_mentions", "follower_count", "is_verified", "has_location", "account_age_days"}
Combined test set from both datasets
Uses binary cross entropy loss and adam


## Reddit Dataset
...

## Model Two Architecture
- Feature Extractors:
- Description: Linear(768 → 8) + LeakyReLU.
- Tweets: Linear(768 → 8) + LeakyReLU.
- Numerical Properties: Linear(5 → 8) + LeakyReLU.
- Categorical Properties: Linear(1 → 8) + LeakyReLU.
- Graph Processing:
- Input Transformation: Linear(32 → 32) + LeakyReLU.
- RGCNConv Layer: Processes graph structure with 2 relation types.
- Output Layers:- Intermediate: 
- Linear(32 → 32) + LeakyReLU.
- Final: Linear(32 → 2) for classification.
Parameter Count:
- Total: 17,650.
- Trainable: 17,650.
Activation Function:
Leaky Relu
Loss Function:
Cross Entropy

## Sources
[https://botometer.osome.iu.edu/bot-repository/datasets.html

https://github.com/warproxxx/Twitter-Bot-or-Not](https://botometer.osome.iu.edu/bot-repository/datasets.html

https://github.com/warproxxx/Twitter-Bot-or-Not

(Model 2) https://github.com/LuoUndergradXJTU/TwiBot-22/

(Model 2) https://github.com/travistangvh/TwitterBotBusters/tree/master/src/BotRGCN

(Cresci-2015) https://arxiv.org/abs/1509.04098)
