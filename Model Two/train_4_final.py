import os
import torch
from torch import nn
from utils import accuracy, init_weights
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from Dataset import Twibot22
from torch_geometric.nn import RGCNConv, FastRGCNConv, GCNConv, GATConv
import torch.nn.functional as F
import pandas as pd


device = 'cuda:0'
embedding_size = 32
dropout = 0.1
lr = 1e-2
weight_decay = 5e-2
epochs = 200
root = './processed_data/'

dataset = Twibot22(root=root, device=device, process=False, save=False)
des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx = dataset.dataloader()

min_users = min(
    des_tensor.size(0),
    tweets_tensor.size(0),
    num_prop.size(0),
    category_prop.size(0),
    labels.size(0)
)
print(f"Truncating to {min_users} users")
des_tensor = des_tensor[:min_users]
tweets_tensor = tweets_tensor[:min_users]
num_prop = num_prop[:min_users]
category_prop = category_prop[:min_users]
labels = labels[:min_users]
train_idx = train_idx[train_idx < min_users]
val_idx   = val_idx[val_idx < min_users]
test_idx  = test_idx[test_idx < min_users]

print(
    f"Shapes → des {des_tensor.shape}, tweets {tweets_tensor.shape},\n"
    f"         num_prop {num_prop.shape}, cat_prop {category_prop.shape}, labels {labels.shape}"
)

class BotRGCN(nn.Module):
    def __init__(self, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, embedding_dimension=32, dropout=0.1):
        super().__init__()
        self.linear_relu_des = nn.Sequential(
            nn.Linear(des_size, embedding_dimension//4),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, embedding_dimension//4),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, embedding_dimension//4),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, embedding_dimension//4),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.rgcn = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        self.dropout = dropout

    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type):
        des      = des.to(self.linear_relu_des[0].weight.device)
        tweet    = tweet.to(self.linear_relu_tweet[0].weight.device)
        num_prop = num_prop.to(self.linear_relu_num_prop[0].weight.device)
        cat_prop = cat_prop.to(self.linear_relu_cat_prop[0].weight.device)

        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x

model      = BotRGCN(cat_prop_size=3, embedding_dimension=embedding_size).to(device)
loss_fn    = nn.CrossEntropyLoss()
optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
model.apply(init_weights)

train_losses, train_accs, val_accs = [], [], []

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output     = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
    loss_train = loss_fn(output[train_idx], labels[train_idx])
    acc_train  = accuracy(output[train_idx], labels[train_idx])
    acc_val    = accuracy(output[val_idx], labels[val_idx])
    loss_train.backward()
    optimizer.step()

    train_losses.append(loss_train.item())
    train_accs.append(acc_train.item())
    val_accs.append(acc_val.item())

    print(
        f'Epoch {epoch+1:03d} | '
        f'loss_train {loss_train:.4f} | '
        f'acc_train {acc_train:.4f} | '
        f'acc_val {acc_val:.4f}'
    )
    return acc_train, loss_train

def test():
    model.eval()
    with torch.no_grad():
        output    = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
        loss_test = loss_fn(output[test_idx], labels[test_idx])
        acc_test  = accuracy(output[test_idx], labels[test_idx])
        preds     = output[test_idx].max(1)[1].cpu().numpy()
        true      = labels[test_idx].cpu().numpy()

        f1        = f1_score(true, preds)
        precision = precision_score(true, preds)
        recall    = recall_score(true, preds)
        probs     = torch.softmax(output[test_idx], dim=1)[:,1].cpu().numpy()
        fpr, tpr, _ = roc_curve(true, probs, pos_label=1)
        Auc       = auc(fpr, tpr)

        print("Test Set Results")
        print(f"Test loss: {loss_test.item():.4f}")
        print(f"Test accuracy: {acc_test.item():.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {Auc:.4f}")


for epoch in range(epochs):
    train(epoch)
test()

plots_dir = './training_plots'
os.makedirs(plots_dir, exist_ok=True)
epochs_list = list(range(1, epochs+1))

plt.figure()
plt.plot(epochs_list, train_losses, label='loss')
plt.plot(epochs_list, val_accs,    label='val acc')
plt.xlabel('epoch'); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'metrics.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(epochs_list, train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(plots_dir, 'training_loss.png')
plt.savefig(loss_plot_path)
plt.close() 

plt.figure(figsize=(10, 6))
plt.plot(epochs_list, train_accs, label='Training Accuracy')
plt.plot(epochs_list, val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
accuracy_plot_path = os.path.join(plots_dir, 'accuracy_plot.png')
plt.savefig(accuracy_plot_path)
plt.close() 

models_dir = './saved_models'
os.makedirs(models_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(models_dir, 'botrgcn_trained_model.pt'))
print("\n Model Saved")
