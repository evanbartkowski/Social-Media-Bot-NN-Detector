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
device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embedding_size  = 32
root_cresci     = './processed_data/'
weights_22      = './saved_models/botrgcn_trained_model.pt'
output_dir      = './finetune_22_to_cresci'
os.makedirs(output_dir, exist_ok=True)

epochs          = 150
warmup_steps   = int(0.1 * epochs)
freeze_epochs   = 3
dropout         = 0.1
lr              = 5e-4
wd              = 5e-2
best_k          = 3    # how many checkpoints to average
patience        = 5    # early stopping on val_loss

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

# matching number of users of twitbo22 to cersci
min_users = min(t.size(0) for t in (desC,tweetsC,numC,catC,labelsC))
print(f"Truncating all feature tensors to first {min_users} users")
desC      = desC[:min_users]
tweetsC   = tweetsC[:min_users]
numC      = numC[:min_users]
catC      = catC[:min_users]
labelsC   = labelsC[:min_users]
trainC    = trainC[trainC < min_users]
valC      = valC[valC < min_users]
testC     = testC[testC < min_users]

# build BotRGCN with cresci-2015 config and load twibot-22 weights
print("Building model and loading Twibot-22 weights")
model = BotRGCN(
    des_size=768, 
    tweet_size=768,
    num_prop_size=numC.size(1), 
    cat_prop_size=1,        # twibot has 3 categorical feature
    embedding_dimension=embedding_size,
    dropout=dropout                   
).to(device)

state = torch.load(weights_22, map_location=device)
# drop mismatched cat_prop params
for k in ('linear_relu_cat_prop.0.weight','linear_relu_cat_prop.0.bias'):
    state.pop(k, None)
model.load_state_dict(state, strict=False)

# freeze graph layers for first few epochs
for name, p in model.named_parameters():
    if 'rgcn' in name:
        p.requires_grad = False

# loss with class‐weights
counts = torch.bincount(labelsC[trainC])
weights = (1.0 / counts.float())
weights = weights / weights.sum() * 2.0
loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

# optimizer & LR scheduler w/ warmup + cosine decay
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
total_steps = epochs
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (epochs - warmup_steps)))
scheduler = LambdaLR(optimizer, lr_lambda)

# training loop
best_states = []
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()

    # unfreeze after freeze_epochs
    if epoch == freeze_epochs+1:
        for name, p in model.named_parameters():
            if 'rgcn' in name:
                p.requires_grad = True

    # full‐graph forward
    out = model(desC, tweetsC, numC, catC, edgeC, typeC)
    loss_train = loss_fn(out[trainC], labelsC[trainC])
    loss_train.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # evaluation
    model.eval()
    with torch.no_grad():
        out_val = model(desC, tweetsC, numC, catC, edgeC, typeC)
        loss_val = loss_fn(out_val[valC], labelsC[valC]).item()
        preds_val = out_val[valC].max(1)[1]
        acc_val = accuracy_score(labelsC[valC].cpu(), preds_val.cpu())

    print(f"Epoch {epoch:02d}  train_loss={loss_train.item():.4f}"
          f"  val_loss={loss_val:.4f}  val_acc={acc_val:.4f}")

    # checkpointing
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # keep top‐k state dicts
    best_states.append((model.state_dict(), loss_val))
    best_states = sorted(best_states, key=lambda x: x[1])[:best_k]

    if epochs_no_improve >= patience:
        print(f"Early stopping after {epoch} epochs")
        break

# average top‐k checkpoints
avg_state = {}
for k in best_states[0][0].keys():
    avg_state[k] = torch.stack([st[0][k].float() for st in best_states], 0).mean(0)
torch.save(avg_state, os.path.join(output_dir, 'best_22_to_cresci.pth'))
print(f"Fine-tuning complete, averaged best-{best_k} saved to {output_dir}/best_22_to_cresci.pth")

# evaluation
model.load_state_dict(avg_state)
model.eval()
with torch.no_grad():
    out_test = model(desC, tweetsC, numC, catC, edgeC, typeC)
    loss_test = loss_fn(out_test[testC], labelsC[testC]).item()
    preds     = out_test[testC].max(1)[1].cpu().numpy()
    true      = labelsC[testC].cpu().numpy()
    probs     = torch.softmax(out_test[testC], dim=1)[:,1].cpu().numpy()
    fpr, tpr, _ = roc_curve(true, probs, pos_label=1)
    auc_score   = auc(fpr, tpr)

# metrics
acc  = accuracy_score(true, preds)
prec = precision_score(true, preds)
rec  = recall_score(true, preds)
f1   = f1_score(true, preds)

print("\n=== Fine-tuned: Twibot-22 → Cresci-2015 ===")
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
plt.xticks([0,1], ['Human','Bot'])
plt.yticks([0,1], ['Human','Bot'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center')
plt.xlabel('Pred'); plt.ylabel('True')
plt.title('Confusion Matrix\n22→Cresci')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_22_to_cresci.png'))
plt.close()

# ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve\n22→Cresci')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve_22_to_cresci.png'))
plt.close()

print(f"\nAll results & plots saved under {output_dir}")
