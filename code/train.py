#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/5/7 8:30
# @Author : jackfrank
# @Version：V 0.1
# @File : train.py
# @desc :
"""
    
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

torch.manual_seed(1)
IF_CUDA = False
if torch.cuda.is_available():
    try:
        IF_CUDA = True
    except Exception as e:  # 防止GPU占用
        print(e)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def trunc_pad(_list: list, max_len=100, if_sentence=True):  # 设置输入句子的最大长度
    if len(_list) == max_len:
        return _list
    if len(_list) > max_len:
        if if_sentence:
            return ["[PAD]"] * 100
        else:

            return ["O"] * 100
    else:
        if if_sentence:
            _list.extend(["[PAD]"] * (max_len - len(_list)))
            return _list
        else:
            _list.extend(["O"] * (max_len - len(_list)))
            return _list


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_bacth(vec):
    max_score_vec = torch.max(vec, dim=1)[0]
    max_score_broadcast = max_score_vec.view(vec.shape[0], -1).expand(vec.shape[0], vec.size()[1])
    return max_score_vec + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self, bacth=1):
        return (torch.randn(2, bacth, self.hidden_dim // 2).cuda(),
                torch.randn(2, bacth, self.hidden_dim // 2).cuda())

    def _forward_alg(self, batchfeats):
        alpha_list = []
        for feats in batchfeats:
            # Do the forward algorithm to compute the partition function
            init_alphas = torch.full((1, self.tagset_size), -10000.)
            # START_TAG has all of the score.
            init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

            # Wrap in a variable so that we will get automatic backprop
            forward_var = init_alphas

            # Iterate through the sentence
            for feat in feats:
                alphas_t = []  # The forward tensors at this timestep
                for next_tag in range(self.tagset_size):
                    # broadcast the emission score: it is the same regardless of
                    # the previous tag
                    emit_score = feat[next_tag].view(
                        1, -1).expand(1, self.tagset_size)
                    # the ith entry of trans_score is the score of transitioning to
                    # next_tag from i
                    trans_score = self.transitions[next_tag].view(1, -1)
                    # The ith entry of next_tag_var is the value for the
                    # edge (i -> next_tag) before we do log-sum-exp
                    next_tag_var = forward_var + trans_score + emit_score
                    # The forward variable for this tag is log-sum-exp of all the
                    # scores.
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            alpha = log_sum_exp(terminal_var)
            alpha_list.append(alpha.view(1))
        return torch.cat(alpha_list)

    def _forward_alg_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        if IF_CUDA:
            init_alphas = torch.full((feats.shape[0], self.tagset_size), -10000.).cuda()
        else:
            init_alphas = torch.full((feats.shape[0], self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas  # [1,6]
        convert_feats = feats.permute(1, 0, 2)
        # Iterate through the sentence
        for feat in convert_feats:  # feat 6
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[:, next_tag].view(
                    feats.shape[0], -1).expand(feats.shape[0], self.tagset_size)  # [1,6]
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).repeat(feats.shape[0], 1)  # [1,6]
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score  # [1,6]
                # The forward variable for this tag is log-sum-exp of all the

                alphas_t.append(log_sum_exp_bacth(next_tag_var))

            forward_var = torch.stack(alphas_t).permute(1, 0)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].view(1, -1).repeat(feats.shape[0], 1)
        alpha = log_sum_exp_bacth(terminal_var)

        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden(bacth=len(sentence))
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        totalsocre_list = []
        for feat, tag in zip(feats, tags):
            totalscore = torch.zeros(1).cuda()
            tag = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).cuda(), tag])
            for i, smallfeat in enumerate(feat):
                totalscore = totalscore + \
                             self.transitions[tag[i + 1], tag[i]] + smallfeat[tag[i + 1]]
            totalscore = totalscore + self.transitions[self.tag_to_ix[STOP_TAG], tag[-1]]
            totalsocre_list.append(totalscore)
        return torch.cat(totalsocre_list)

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _viterbi_decode_predict(self, feats_list):
        path_list = []
        for feats in feats_list:
            backpointers = []

            # Initialize the viterbi variables in log space
            init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
            init_vvars[0][self.tag_to_ix[START_TAG]] = 0

            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = init_vvars
            for feat in feats:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step

                for next_tag in range(self.tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning
                    # from tag i to next_tag.
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]  # Sanity check
            best_path.reverse()
            path_list.append(best_path)
        return path_list

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg_parallel(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def predict(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        tag_seq_list = self._viterbi_decode_predict(lstm_feats)
        return tag_seq_list


filename = 'data/source_BIO_2014_cropus.txt'
source_data = []
with open(filename, 'r', encoding='UTF-8')as txt_file:
    lines = txt_file.readlines()
    for line in lines:
        source_data.append(line.split())

filename = 'data/target_BIO_2014_cropus.txt'
target_data = []
with open(filename, 'r', encoding='UTF-8')as txt_file:
    lines = txt_file.readlines()
    for line in lines:
        target_data.append(line.split())

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 128
HIDDEN_DIM = 100
BATCH_SIZE = 32  # 批训练的数据个数
num_epochs = 1

word_set = source_data
label_set = target_data

tag_to_ix = {"B_T": 0, "B_LOC": 1, "B_ORG": 2, "B_PER": 3,
             "I_T": 4, "I_LOC": 5, "I_ORG": 6, "I_PER": 7,
             "O": 8, START_TAG: 9, STOP_TAG: 10, "[PAD]": 11}

word_to_ix = {"[PAD]": 11}
for item_st in range(len(word_set)):
    word_set[item_st] = trunc_pad(word_set[item_st], if_sentence=True)
    label_set[item_st] = trunc_pad(label_set[item_st], if_sentence=False)
for sentence in word_set:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

for i in range(len(word_set)):
    word_set[i] = [word_to_ix[t] for t in word_set[i]]
    label_set[i] = [tag_to_ix[t] for t in label_set[i]]

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(torch.tensor(word_set, dtype=torch.long), torch.tensor(label_set, dtype=torch.long))

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  #
    num_workers=2,  # 多线程来读数据
)

# Make up some training data
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(num_epochs):
    for step, (batch_x, batch_y) in enumerate(loader):
        print(str(epoch) + '/' + str(num_epochs) + ':' + ' ' + str(step) + '/' + str(len(word_set) // BATCH_SIZE))
        model.zero_grad()
        loss = model.neg_log_likelihood(torch.tensor(batch_x, dtype=torch.long).cuda(),
                                        torch.tensor(batch_y, dtype=torch.long).cuda())
        print(loss)
        loss.backward()
        optimizer.step()

torch.save(model, 'bilstm_crf.pkl')
# model = torch.load('Model/bilstm_crf.pkl')
# Check predictions after training
with torch.no_grad():
    precheck_sent = torch.tensor([word_set[0], word_set[1]], dtype=torch.long).cuda()
    print(label_set[0], label_set[1])
    print(model.predict(precheck_sent.view(len(precheck_sent), -1)))

