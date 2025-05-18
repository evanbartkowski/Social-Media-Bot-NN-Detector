#!/usr/bin/env python3
import os, random, math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
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
root_twibot22  = './processed_data/'
weights_15     = './saved_models/15_BotRGCN_weight.pth'
output_dir     = './finetune_15_to_22_v2'
os.makedirs(output_dir, exist_ok=True)

num_epochs     = 100
warmup_steps   = int(0.1 * num_epochs)
max_grad_norm  = 1.0
lr             = 1e-4
wd             = 1e-3
dropout        = 0.1
gamma_focal    = 2.0

# focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma * ce)
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

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

# matching number of users of cersci to twitbo22
min_u = min(t.size(0) for t in (des22,tweets22,num22,cat22,labels22))
print(f"Truncating all feature tensors to first {min_u} users")
des22    = des22[:min_u]
tweets22 = tweets22[:min_u]
num22    = num22[:min_u]
cat22    = cat22[:min_u]
labels22 = labels22[:min_u]
train22  = train22[train22<min_u]
val22    = val22[val22<min_u]
test22   = test22[test22<min_u]

# build BotRGCN with twibot-22 config and load cresci-2015 weights
print("Building BotRGCN and loading Cresci-15 weights")
model = BotRGCN(
    des_size=768, 
    tweet_size=768,
    num_prop_size=num22.size(1), 
    cat_prop_size=3,            # twibot has 3 categorical feature
    embedding_dimension=embedding_size,
    dropout=dropout
).to(device)

state = torch.load(weights_15, map_location=device)
# zero-shot inference on twibot-22
for k in ('linear_relu_cat_prop.0.weight','linear_relu_cat_prop.0.bias'):
    state.pop(k, None)
model.load_state_dict(state, strict=False)

# all layers trainable
for p in model.parameters():
    p.requires_grad = True

# compute class weights for focalloss
counts = torch.bincount(labels22[train22])
alpha = (1.0 / counts.float())
alpha = alpha / alpha.sum() * 2.0
loss_fn = FocalLoss(alpha=alpha.to(device), gamma=gamma_focal)

# optimizer and scheduleing
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (num_epochs - warmup_steps)))
scheduler = LambdaLR(optimizer, lr_lambda)

# training loop
best_val_loss = float('inf')
best_state = None
for epoch in range(1, num_epochs+1):
    model.train()
    optimizer.zero_grad()

    # full‐graph forward
    out = model(des22, tweets22, num22, cat22, edge22, type22)
    loss_train = loss_fn(out[train22], labels22[train22])
    loss_train.backward()
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()

    # evaluation
    model.eval()
    with torch.no_grad():
        out_val = model(des22, tweets22, num22, cat22, edge22, type22)
        loss_val = loss_fn(out_val[val22], labels22[val22]).item()
        pred_val = out_val[val22].argmax(1)
        acc_val  = accuracy_score(labels22[val22].cpu(), pred_val.cpu())

    print(f"Epoch {epoch:02d}  train_loss={loss_train.item():.4f}"
          f"  val_loss={loss_val:.4f}  val_acc={acc_val:.4f}")

    # checkpointing
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        best_state = model.state_dict()

# save best
torch.save(best_state, os.path.join(output_dir,'best_15_to_22_v2.pth'))
print("Saved best checkpoint.")

# evaluation
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out_t = model(des22, tweets22, num22, cat22, edge22, type22)
    loss_test = loss_fn(out_t[test22], labels22[test22]).item()
    preds    = out_t[test22].argmax(1).cpu().numpy()
    true     = labels22[test22].cpu().numpy()
    probs    = torch.softmax(out_t[test22],1)[:,1].cpu().numpy()
    fpr,tpr,_ = roc_curve(true,probs,pos_label=1)
    auc_score = auc(fpr,tpr)

# metrics
acc   = accuracy_score(true, preds)
prec  = precision_score(true, preds)
rec   = recall_score(true, preds)
f1    = f1_score(true, preds)

print("\n=== Fine-tuned v2: Cresci-15 → Twitbot-22 ===")
print(f"Loss     : {loss_test:.4f}")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc_score:.4f}")

# confusion matrix
cm = confusion_matrix(true, preds)
plt.figure(figsize=(4,4))
plt.imshow(cm, interpolation='nearest')
plt.xticks([0,1],['Human','Bot']); plt.yticks([0,1],['Human','Bot'])
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha='center',va='center')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix\n15→22 v2')
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'confusion_matrix_15_to_22_v2.png'))
plt.close()

# ROC curve
plt.figure()
plt.plot(fpr,tpr,label=f'AUC = {auc_score:.3f}')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve\n15→22 v2')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'roc_curve_15_to_22_v2.png'))
plt.close()

print(f"\nAll results & plots saved under {output_dir}")
