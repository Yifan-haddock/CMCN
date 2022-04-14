from torch.utils import data
from model import em_model
import joblib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

with open('biosyn_disorder_query_disorder_dictionary_train_sample.joblib','rb') as f:
    dataset = joblib.load(f)

sparse_score = torch.tensor(dataset['tfidf_scores'])[20000:30000,:]
dense_score = torch.tensor(dataset['bert_scores'])[20000:30000,:]
labels = torch.tensor(dataset['labels'])[20000:30000,:]

# with open('sapbert_icd10_query_disorder_dictionary_train_sample.joblib','rb') as f:
#     dataset2 = joblib.load(f)

# sparse_score2 = torch.tensor(dataset2['tfidf_scores'])
# dense_score2 = torch.tensor(dataset2['bert_scores'])
# labels2 = torch.tensor(dataset2['labels'])

# with open('sapbert_realworld_query_disorder_dictionary_train_sample.joblib','rb') as f:
#     dataset3 = joblib.load(f)

# sparse_score3 = torch.tensor(dataset3['tfidf_scores'])
# dense_score3 = torch.tensor(dataset3['bert_scores'])
# labels3 = torch.tensor(dataset3['labels'])

# sparse_score = torch.cat([sparse_score,sparse_score2],dim = 0)
# dense_score = torch.cat([dense_score,dense_score2],dim = 0)
# labels = torch.cat([labels,labels2],dim = 0)

tensordataset = TensorDataset(dense_score,sparse_score,labels)
# train_tensordataset,test_tensordataset = train_test_split(tensordataset,test_size=0.7,shuffle=True)
train_tensordataset = tensordataset

dataloader = DataLoader(train_tensordataset, batch_size=64, shuffle= True)

model = em_model(dense_weight = 1.0, learning_rate=0.01)

def train(dataloader, model):
    train_loss = 0
    train_steps = 0
    for data in dataloader:
        model.optimizer.zero_grad()
        dense, sparse, labels = data
        x = dense, sparse
        scores = model(x)
        loss = model.loss(scores, labels)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1
        torch.cuda.empty_cache()
    train_loss /= (train_steps + 1e-9)
    return train_loss


for epoch in range(1,501):
    trainloss = train(dataloader, model)
    if epoch % 5 == 0:
        print('Epoch %d, Loss %f' % (epoch, float(trainloss)))
        print(model.__param__())

sparse_weight_trained = model.__param__()
print(sparse_weight_trained)
with open('sapbert_sparse_weight.bin', 'wb') as f:
    joblib.dump(sparse_weight_trained, f)