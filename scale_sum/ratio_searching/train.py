from torch.utils import data
from model import em_model
import joblib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n','--nsamples',default=100,type=int,required=True)
parser.add_argument('--pretrained')
parser.add_argument('--epoch',default=501,type=int,required=True)
parser.add_argument('--times')
args = parser.parse_args()

with open(f'{args.pretrained}_disorder_query_disorder_dictionary_train_sample.joblib','rb') as f:
    dataset = joblib.load(f)

sparse_score = torch.tensor(dataset['tfidf_scores'])
dense_score = torch.tensor(dataset['bert_scores'])
labels = torch.tensor(dataset['labels'])

def random_select_dataset(sparse_score, dense_score, labels, n =100):
    random_index = torch.randint(len(sparse_score),(n,))
    return sparse_score[random_index], dense_score[random_index], labels[random_index]

sparse_score, dense_score, labels = random_select_dataset(sparse_score, dense_score, labels, n =args.nsamples)

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


for epoch in range(1,args.epoch):
    trainloss = train(dataloader, model)
    if epoch % 5 == 0:
        print('Epoch %d, Loss %f' % (epoch, float(trainloss)))
        print(model.__param__()[0]/model.__param__()[1])

sparse_weight_trained = model.__param__()
print(sparse_weight_trained[0]/sparse_weight_trained[1])
# with open(f'weight_training/{args.pretrained}_n{str(args.nsamples)}_times{args.times}_sparse_weight.bin', 'wb') as f:
#     joblib.dump(sparse_weight_trained, f)