#!/usr/bin/env python3
import os, random, numpy as np
import torch, torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix
)
import matplotlib.pyplot as plt
from model import BotRGCN

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# config
device            = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embedding_size    = 32
root_twibot22     = './processed_data'               # Twibot-22 folder
weights_cresci15  = './saved_models/15_BotRGCN_weight.pth'
output_dir        = './zero_shot_cresci_to_22'
os.makedirs(output_dir, exist_ok=True)

# helper to load .pt
def load22(name):
    return torch.load(os.path.join(root_twibot22, name), map_location=device)

print("Loading Twibot-22 tensors…")
des22      = load22('des_tensor.pt')
tweets22   = load22('tweets_tensor.pt')
num22      = load22('num_properties_tensor.pt')
cat22      = load22('cat_properties_tensor.pt')
edge22     = load22('edge_index.pt')
type22     = load22('edge_type.pt')
labels22   = load22('label.pt')
train22    = load22('train_idx.pt')
val22      = load22('val_idx.pt')
test22     = load22('test_idx.pt')
print("Done.")

# matching number of users of cersci to twitbo22
min_u = min(t.size(0) for t in (des22,tweets22,num22,cat22,labels22))
des22    = des22[:min_u]
tweets22 = tweets22[:min_u]
num22    = num22[:min_u]
cat22    = cat22[:min_u]
labels22 = labels22[:min_u]
train22  = train22[train22<min_u]
val22    = val22[val22<min_u]
test22   = test22[test22<min_u]
print(f"Truncated all features to {min_u} users")

# build BotRGCN with twibot-22 config and load cresci-2015 weights
print("Building model (cat_prop_size=3) and loading Cresci-2015 weights")
model = BotRGCN(
    des_size=768,
    tweet_size=768,
    num_prop_size=num22.size(1),
    cat_prop_size=3,               # twibot-22 has 3 categorical features
    embedding_dimension=embedding_size,
    dropout=0.1
).to(device)

state = torch.load(weights_cresci15, map_location=device)
# strip out cresci-vs-twibot mismatch on cat_prop layers
for key in ('linear_relu_cat_prop.0.weight','linear_relu_cat_prop.0.bias'):
    state.pop(key, None)

model.load_state_dict(state, strict=False)
model.eval()

# zero-shot inference on twibot-22
loss_fn = nn.CrossEntropyLoss()
with torch.no_grad():
    out       = model(des22, tweets22, num22, cat22, edge22, type22)
    loss_test = loss_fn(out[test22], labels22[test22]).item()
    preds     = out[test22].max(1)[1].cpu().numpy()
    true      = labels22[test22].cpu().numpy()
    probs     = torch.softmax(out[test22], dim=1)[:,1].cpu().numpy()
    fpr, tpr, _ = roc_curve(true, probs, pos_label=1)
    auc_score   = auc(fpr, tpr)

# metrics
acc   = accuracy_score(true, preds)
prec  = precision_score(true, preds)
rec   = recall_score(true, preds)
f1    = f1_score(true, preds)

print("\n=== Zero-Shot: Cresci-2015 → Twibot-22 ===")
print(f"Loss     : {loss_test:.4f}")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc_score:.4f}")

# confusion matrix
cm = confusion_matrix(true, preds)
fig, ax = plt.subplots()
ax.imshow(cm, interpolation='nearest')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Human','Bot']); ax.set_yticklabels(['Human','Bot'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha='center', va='center')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix\nCresci→Twibot22')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'confusion_matrix_cresci_to_22.png'))
plt.close()

# ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve\nCresci→Twibot22')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'roc_curve_cresci_to_22.png'))
plt.close()

print(f"\nResults & plots saved under {output_dir}")
