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
device         = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embedding_size = 32
root_cresci    = './processed_data/'
weights_22     = './saved_models/botrgcn_trained_model.pt'
output_dir     = './zero_shot_22_to_cresci'
os.makedirs(output_dir, exist_ok=True)

# helper to load .pt
def load_pt(name):
    return torch.load(os.path.join(root_cresci, name), map_location=device)

print("Loading Cresci-2015 tensors…")
desC    = load_pt('des_tensor.pt')
tweetsC = load_pt('tweets_tensor.pt')
numC    = load_pt('num_properties_tensor.pt')
catC    = load_pt('cat_properties_tensor.pt')
edgeC   = load_pt('edge_index.pt')
typeC   = load_pt('edge_type.pt')
labelsC = load_pt('label.pt')
trainC  = load_pt('train_idx.pt')
valC    = load_pt('val_idx.pt')
testC   = load_pt('test_idx.pt')
print("Done.")

# build BotRGCN with cresci-2015 config and load twibot-22 weights
print("Building model (cat_prop_size=1) and loading Twibot-22 weights")
model = BotRGCN(
    des_size=768,
    tweet_size=768,
    num_prop_size=numC.size(1),
    cat_prop_size=1,              # cresci has 1 categorical feature
    embedding_dimension=embedding_size,
    dropout=0.1
).to(device)

state = torch.load(weights_22, map_location=device)
# strip out cresci-vs-twibot mismatch on cat_prop layers
for key in ('linear_relu_cat_prop.0.weight', 'linear_relu_cat_prop.0.bias'):
    state.pop(key, None)

model.load_state_dict(state, strict=False)
model.eval()

# zero-shot inference on cresci-2015
loss_fn = nn.CrossEntropyLoss()
with torch.no_grad():
    out       = model(desC, tweetsC, numC, catC, edgeC, typeC)
    loss_test = loss_fn(out[testC], labelsC[testC]).item()
    preds     = out[testC].max(1)[1].cpu().numpy()
    true      = labelsC[testC].cpu().numpy()
    probs     = torch.softmax(out[testC], dim=1)[:,1].cpu().numpy()
    fpr, tpr, _ = roc_curve(true, probs, pos_label=1)
    auc_score   = auc(fpr, tpr)

# metrics
acc  = accuracy_score(true, preds)
prec = precision_score(true, preds)
rec  = recall_score(true, preds)
f1   = f1_score(true, preds)

print("\n=== Zero-Shot: Twibot-22 → Cresci-2015 ===")
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
ax.set_title('Confusion Matrix\n22→Cresci')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_22_to_cresci.png'))
plt.close()

# ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve\n22→Cresci')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve_22_to_cresci.png'))
plt.close()

print(f"\nResults & plots saved under {output_dir}")
