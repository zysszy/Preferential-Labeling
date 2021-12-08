import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import numpy as np

random_seed = 0
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)


from torch.utils.data import DataLoader
from torch.optim import Adam
from data_loader import data_slice, GraphDataset, GraphCollateFunction
from model import GCNClassifier
from tqdm import tqdm
import logging


NUM_CLASSES = 2

class GraphModel:
    def __init__(self):
        batch_size = 32

        self.logger = logging.getLogger("GraphModel")

        data_train, data_validation, data_test = data_slice(GraphDataset("mis.txt"))
        collate_fn = GraphCollateFunction(4096)
        self.data_train = DataLoader(data_train, batch_size, True, collate_fn=collate_fn, num_workers=3)
        self.data_validation = DataLoader(data_validation, batch_size, False, collate_fn=collate_fn)
        self.data_test = DataLoader(data_test, batch_size, False, collate_fn=collate_fn)
        self.model = GCNClassifier(4096, 128, NUM_CLASSES, 20)
        self.model = self.model.cuda()
        self.optim = Adam(self.model.parameters(), lr=1e-4)
        self.best_acc = 0
        self.step = 0

        self.num_samples_train = 10
        self.num_samples_eval = 10

    def gen_random_shuffle(self, num_nodes, max_num_nodes):
        batch_size = num_nodes.shape[0]
        input_ = torch.zeros([batch_size, max_num_nodes], dtype=torch.long)
        for i, cur_num_nodes in enumerate(num_nodes):
            cur_num_nodes = cur_num_nodes.item()
            input_[i, :cur_num_nodes] = torch.randperm(cur_num_nodes)
        return input_

    def train(self):
        t = tqdm(self.data_train, total=len(self.data_train), desc=f"Train {self.step}")
        self.model.train(True)
        for input_, adjacent, adjacent_values, dense_shape, num_nodes, padding_mask, labels in t:
            adjacent = torch.sparse_coo_tensor(adjacent, adjacent_values, dense_shape)
            adjacent, num_nodes, padding_mask, labels = adjacent.cuda(), num_nodes.cuda(), padding_mask.cuda(), labels.cuda()
            with torch.no_grad():
                min_losses = None
                min_losses_labels = None
                for _ in range(self.num_samples_train):
                    input_ = self.gen_random_shuffle(num_nodes, 4096)
                    input_gpu = input_.cuda()
                    _, loss, _ = self.model(input_gpu, adjacent, num_nodes, padding_mask, labels)
                    loss = loss.cpu().numpy()
                    #print (loss)
                    input_ = input_.numpy()
                    if min_losses is None:
                        min_losses = loss
                        min_losses_labels = input_
                    else:
                        for i, (cur_loss, cur_input) in enumerate(zip(loss, input_)):
                            if cur_loss < min_losses[i]:
                                min_losses[i] = cur_loss
                                min_losses_labels[i] = cur_input
            input_ = torch.LongTensor(min_losses_labels).cuda()
            _, loss, acc = self.model(input_, adjacent, num_nodes, padding_mask, labels)
            loss = torch.mean(loss)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.step += 1
            train_info = f"Train {self.step} {loss.item():08.06f} {acc.item()*100:06.03f}%"
            t.set_description(train_info)
            self.logger.debug(train_info)

    def validation(self):
        t = tqdm(self.data_validation, total=len(self.data_validation), desc=f"Validation")
        total_num = 0
        total_correct = 0
        resolvedMIS = 0
        for input_, adjacent, adjacent_values, dense_shape, num_nodes, padding_mask, labels in t:
            with torch.no_grad():
                self.model.train(False)
                adj = adjacent
                adjacent = torch.sparse_coo_tensor(adjacent, adjacent_values, dense_shape)
                adjacent, num_nodes, padding_mask, labels = adjacent.cuda(), num_nodes.cuda(), padding_mask.cuda(), labels.cuda()
                pred_likehood_final = None
                pred_label_final = None
                pred_lik = None
                for _ in range(self.num_samples_eval):
                    input_ = self.gen_random_shuffle(num_nodes, 4096)
                    input_ = input_.cuda()
                    pred = self.model(input_, adjacent, num_nodes, padding_mask)
                    pred = torch.softmax(pred, -1).masked_fill(~padding_mask.unsqueeze(-1), 1)
                    pred_likehood, pred_label = torch.max(pred, -1)
                    lik = pred.cpu().numpy()[:,:,1]
                    pred_likehood = np.prod(pred_likehood.cpu().numpy(), -1)
                    pred_label = pred_label.cpu().numpy()
                    if pred_likehood_final is None:
                        pred_likehood_final = pred_likehood
                        pred_label_final = pred_label
                        pred_lik = lik
                    else:
                        for i, (cur_likehood, lk, cur_label) in enumerate(zip(pred_likehood, lik, pred_label)):
                            if cur_likehood > pred_likehood_final[i]:
                                pred_likehood_final[i] = cur_likehood
                                pred_lik[i] = lk
                                pred_label_final[i] = cur_label
                labels = labels.cpu().numpy()
                #acc = np.average((labels == pred_label_final).astype(float))
                resolvedMIS += self.checkMIS(pred_lik, labels, pred_label_final, padding_mask.cpu().numpy().astype(float), adj.cpu().numpy())
                acc = np.sum((labels == pred_label_final).astype(float)*padding_mask.cpu().numpy().astype(float))/np.sum(padding_mask.cpu().numpy().astype(float))
                acc = float(acc)
                t.set_description(f"Validation {acc*100:06.03f}%")
                total_num += labels.shape[0]
                total_correct += labels.shape[0] * acc
        acc = total_correct / total_num
        isbest = False
        if acc > self.best_acc:
            self.best_acc = acc
            os.makedirs("train", exist_ok=True)
            self.save("train/best.model")
            isbest = True
        self.logger.info(f"Evaluation {self.step} {self.best_acc*100:06.03f}%  Solve: {resolvedMIS} {'BEST' if isbest else ''} ")

    def save(self, path):
        torch.save({"model": self.model.state_dict(), "optim": self.optim.state_dict(), "step": self.step, "best_acc": self.best_acc}, path)
        self.logger.info(f"Model saved to {path}")

    def load_if_exists_best(self):
        if os.path.exists("train/best.model"):
            self.load("train/best.model")

    def load(self, path):
        d = torch.load(path)
        self.model.load_state_dict(d["model"])
        self.optim.load_state_dict(d["optim"])
        self.step = d["step"]
        self.best_acc = d["best_acc"]
        self.logger.info(f"Model loaded from {path}")

    def adj2list (self, adj):
        l = []
        start = 0
        end = -1
        for i in range(len(adj[0])):
            if i == len(adj[0]) - 1 or adj[0][i] != adj[0][i + 1]:
                end = i + 1
                l2 = [[] for _ in range(100000)]
                for k in range(start, end):
                    l2[int(adj[1][k])].append(int(adj[2][k]))
                l.append(l2)
                start = end
        return l

    def checkMIS(self, likehood, labels, preds, masks, adjold):
        indexes = np.argsort(likehood, axis=-1)[:,::-1]
        assert np.shape(indexes) == np.shape(likehood)
        #print (np.shape(indexes))
        length = np.sum(masks, -1).astype(int)
        count = 0
        adj = self.adj2list(adjold)
        assert len(adj) == len(indexes)
        numbers = np.sum(labels, -1)
        for i in range(len(indexes)):
            l = [0] * length[i]
            for t in range(len(indexes[i])):
                index = indexes[i][t]
                #print (index)
                if index < length[i]:
                    #print (index, length[i])
                    #print ("enter")
                    flag = 1
                    for k in range(len(adj[i][index])):
                        if l[int(adj[i][index][k])] == 1:
                            flag = 0
                            break
                    l[index] = flag

            if sum(l) > numbers[i]:
                assert False
            elif sum(l) == numbers[i]:
                count += 1
            #count += numbers[i] - sum(l)
            #print (sum(l), numbers[i])
        return count

    def test(self):
        t = tqdm(self.data_test, total=len(self.data_test), desc=f"Test")
        total_num = 0
        total_correct = 0
        resolvedMIS = 0
        for input_, adjacent, adjacent_values, dense_shape, num_nodes, padding_mask, labels in t:
            with torch.no_grad():
                self.model.train(False)
                adj = adjacent
                adjacent = torch.sparse_coo_tensor(adjacent, adjacent_values, dense_shape)
                adjacent, num_nodes, padding_mask, labels = adjacent.cuda(), num_nodes.cuda(), padding_mask.cuda(), labels.cuda()
                pred_likehood_final = None
                pred_label_final = None
                pred_lik = None
                for _ in range(self.num_samples_eval):
                    input_ = self.gen_random_shuffle(num_nodes, 4096)
                    input_ = input_.cuda()
                    pred = self.model(input_, adjacent, num_nodes, padding_mask)
                    pred = torch.softmax(pred, -1).masked_fill(~padding_mask.unsqueeze(-1), 1)
                    pred_likehood, pred_label = torch.max(pred, -1)
                    lik = pred.cpu().numpy()[:,:,1]
                    pred_likehood = np.prod(pred_likehood.cpu().numpy(), -1)
                    pred_label = pred_label.cpu().numpy()
                    if pred_likehood_final is None:
                        pred_likehood_final = pred_likehood
                        pred_lik = lik
                        pred_label_final = pred_label
                    else:
                        for i, (cur_likehood, lk, cur_label) in enumerate(zip(pred_likehood, lik, pred_label)):
                            if cur_likehood > pred_likehood_final[i]:
                                pred_likehood_final[i] = cur_likehood
                                pred_lik[i] = lk
                                pred_label_final[i] = cur_label
                labels = labels.cpu().numpy()
                resolvedMIS += self.checkMIS(pred_lik, labels, pred_label_final, padding_mask.cpu().numpy().astype(float), adj.cpu().numpy())
                acc = np.sum((labels == pred_label_final).astype(float)*padding_mask.cpu().numpy().astype(float))/np.sum(padding_mask.cpu().numpy().astype(float))
                acc = float(acc)
                t.set_description(f"Test {acc*100:06.03f}%")
                total_num += labels.shape[0]
                total_correct += labels.shape[0] * acc
        acc = total_correct / total_num
        self.logger.info(f"Test result: {acc*100:06.03f} {resolvedMIS}")
        return acc


def main():
    os.makedirs("train", exist_ok=True)
    logging.basicConfig(
        filename="train/train.log", level=logging.DEBUG, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    logging.getLogger().info("Program Started!")

    graphmodel = GraphModel()
    graphmodel.load_if_exists_best()
#    graphmodel.test()
#    exit()
    count = 0
    while True:
        graphmodel.train()
        graphmodel.validation()
        count += 1
#        graphmodel.test()


if __name__ == "__main__":
    main()
