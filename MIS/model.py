import torch
from torch.nn import Module, Linear, LayerNorm, ModuleList, Embedding, Dropout
from torch.nn.functional import relu, log_softmax, nll_loss, softmax


class GCNLayer(Module):
    def __init__(self, in_features, dropout_rate=0.1):
        super().__init__()

        self.msg_linear1 = Linear(in_features, in_features)
        self.msg_linear2 = Linear(in_features, in_features)
        self.msg_linear3 = Linear(in_features, in_features)
        self.msg_linear4 = Linear(in_features, in_features)

        self.layer_norm = LayerNorm([in_features])
        self.msg_dropout = Dropout(dropout_rate)

    # input_ [batch_size, num_nodes, in_features]
    # adjacent [batch_size, num_nodes_to, num_nodes_from]
    def forward(self, input_, adjacent):
        msg = self.msg_linear1(input_)
        msg = relu(msg)
        msg = self.msg_linear2(input_)
        msg = torch.bmm(adjacent, msg)
        msg = self.msg_linear3(msg)
        msg = relu(msg)
        msg = self.msg_linear4(msg)
        msg = self.msg_dropout(msg)
        input_ = input_ + msg
        input_ = self.layer_norm(input_)
        return input_


class GCNClassifier(Module):
    def __init__(self, in_classes, hidden_features, out_features, layers):
        super().__init__()

        self.embedding = Embedding(in_classes, hidden_features)
        self.gcns = ModuleList([GCNLayer(hidden_features) for _ in range(layers)])
        self.out_linear1 = Linear(hidden_features, hidden_features)
        self.out_linear2 = Linear(hidden_features, out_features)

    def forward(self, input_, adjacent, num_nodes, padding_mask, labels=None):
        x = self.embedding(input_)
        for gcn in self.gcns:
            x = gcn(x, adjacent)
        x = x.masked_fill(~padding_mask.unsqueeze(2), 0)
        #x = torch.sum(x, 1) / torch.sum(padding_mask.float(), dim=1).unsqueeze(1)
        x = self.out_linear1(x)
        x = torch.nn.functional.dropout(x, 0.5)
        x = relu(x)
        x = self.out_linear2(x)
        x = x.masked_fill(~padding_mask.unsqueeze(2), 0)
        pred = x

        if labels is None:
            return pred

        x = log_softmax(x, -1).clip(min=-1e6)
        x = x.masked_fill(~padding_mask.unsqueeze(2), 0)
        #print (x.size())
        #print (labels.size())
        #print (padding_mask.size())
        loss = torch.sum(nll_loss(x.view(-1, 2), labels.view(-1), reduction='none').view(padding_mask.shape).masked_fill(~padding_mask, 0), dim=-1) / torch.sum(padding_mask.float(), dim=-1)
#        loss = - torch.sum(torch.gather(x, -1, labels.unsqueeze(-1)).squeeze(-1).masked_fill(~padding_mask, 0), 1) / torch.sum(padding_mask.float(), dim=1)#nll_loss(x, labels, reduction="none")

        acc = torch.argmax(pred, -1) == labels
        acc = acc.masked_fill(~padding_mask, 0)
        acc = torch.sum(acc, [0, 1]) / torch.sum(padding_mask.float(), dim=[0,1])
        #acc = torch.mean(acc.float())

        return pred, loss, acc

