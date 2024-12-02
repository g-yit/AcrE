from helper import *


def interleave_two_tensor(tensor_a, tensor_b, mode='row', first='a'):
    """
    将两个张量按指定方式交替合并，支持带或不带 batch_size 维度。

    参数：
    - tensor_a: PyTorch 张量，形状为 [batch_size, N, M] 或 [N, M]
    - tensor_b: PyTorch 张量，形状与 matrix_a 相同
    - mode: str, 'row' 或 'column'，决定是按行还是按列交替合并

    返回：
    - result: 交替合并后的张量，形状为 [batch_size, 2*N, M] 或 [2*N, M] (行交替)
              或 [batch_size, N, 2*M] 或 [N, 2*M] (列交替)
    """
    # 检查输入维度是否一致
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("两个输入张量的形状必须相同")

    # 如果输入没有 batch_size 维度，增加一个维度
    if tensor_a.dim() == 2:
        tensor_a = tensor_a.unsqueeze(0)  # 形状变为 [1, N, M]
        tensor_b = tensor_b.unsqueeze(0)
        added_batch_dim = True
    else:
        added_batch_dim = False

    batch_size, num_rows, num_cols = tensor_a.size()

    if mode == 'row':
        # 行交替合并，初始化结果张量，形状为 [batch_size, 2 * num_rows, num_cols]
        result = torch.empty((batch_size, num_rows * 2, num_cols), dtype=tensor_a.dtype)
        # 从0开始每隔1个元素放一行matrix_a
        if first == 'a':
            result[:, 0::2] = tensor_a  # 将 matrix_a 的行放在偶数行
            result[:, 1::2] = tensor_b  # 将 matrix_b 的行放在奇数行
        else:
            result[:, 0::2] = tensor_b  # 将 matrix_a 的行放在偶数行
            result[:, 1::2] = tensor_a  # 将 matrix_b 的行放在奇数行

    elif mode == 'column':
        # 列交替合并，初始化结果张量，形状为 [batch_size, num_rows, 2 * num_cols]
        result = torch.empty((batch_size, num_rows, num_cols * 2), dtype=tensor_a.dtype)
        if first == 'a':
            result[:, :, 0::2] = tensor_a  # 将 matrix_a 的列放在偶数列
            result[:, :, 1::2] = tensor_b  # 将 matrix_b 的列放在奇数列
        else:
            result[:, :, 0::2] = tensor_b  # 将 matrix_a 的列放在偶数列
            result[:, :, 1::2] = tensor_a  # 将 matrix_b 的列放在奇数列

    else:
        raise ValueError("mode 参数只能是 'row' 或 'column'")

    # 如果原始输入没有 batch_size 维度，移除该维度
    if added_batch_dim:
        result = result.squeeze(0)  # 形状变回 [2*N, M] 或 [N, 2*M]

    return result


class AcrE(torch.nn.Module):
    """
    Proposed method in the paper. Refer Section 6 of the paper for mode details

    Parameters
    ----------
    params:        	Hyperparameters of the model

    Returns
    -------
    The AcrE model instance

    """

    def __init__(self, params, chequer_perm):
        super(AcrE, self).__init__()

        self.p = params
        self.ent_embed = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim, padding_idx=None);
        xavier_normal_(self.ent_embed.weight)
        self.rel_embed = torch.nn.Embedding(self.p.num_rel * 2, self.p.embed_dim, padding_idx=None);
        xavier_normal_(self.rel_embed.weight)
        self.bceloss = torch.nn.BCELoss()

        self.inp_drop = torch.nn.Dropout(self.p.inp_drop)
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.feat_drop)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.channel)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.fc = torch.nn.Linear(self.p.channel * 400, self.p.embed_dim)
        self.padding = 0
        self.way = self.p.way
        self.first_atrous = self.p.first_atrous
        self.second_atrous = self.p.second_atrous
        self.third_atrous = self.p.third_atrous
        self.chequer_perm = chequer_perm
        if self.way == 's':
            self.conv1 = torch.nn.Conv2d(1, self.p.channel, (3, 3), 1, self.first_atrous, bias=self.p.bias,
                                         dilation=self.first_atrous)
            self.conv2 = torch.nn.Conv2d(self.p.channel, self.p.channel, (3, 3), 1, self.second_atrous,
                                         bias=self.p.bias, dilation=self.second_atrous)
            self.conv3 = torch.nn.Conv2d(self.p.channel, self.p.channel, (3, 3), 1, self.third_atrous, bias=self.p.bias,
                                         dilation=self.third_atrous)
        else:
            self.conv1 = torch.nn.Conv2d(2, self.p.channel, (3, 3), 1, self.first_atrous, bias=self.p.bias,
                                         dilation=self.first_atrous, padding_mode='circular')
            self.conv2 = torch.nn.Conv2d(2, self.p.channel, (3, 3), 1, self.second_atrous, bias=self.p.bias,
                                         dilation=self.second_atrous, padding_mode='circular')
            self.conv3 = torch.nn.Conv2d(2, self.p.channel, (3, 3), 1, self.third_atrous, bias=self.p.bias,
                                         dilation=self.third_atrous, padding_mode='circular')
            self.W_gate_e = torch.nn.Linear(1600, 400)

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

    def loss(self, pred, true_label=None, sub_samp=None):
        label_pos = true_label[0];
        label_neg = true_label[1:]
        loss = self.bceloss(pred, true_label)
        return loss

    def forward(self, sub, rel, neg_ents, strategy='one_to_x'):
        sub_emb = self.ent_embed(sub)
        rel_emb = self.rel_embed(rel)

        alt_sub_emb = sub_emb.reshape(-1, 1, 10, 20)
        alt_rel_emb = rel_emb.reshape(-1, 1, 10, 20)
        alt_comb = interleave_two_tensor(alt_sub_emb, alt_rel_emb)

        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.p.perm, 2 * self.p.k_w, self.p.k_h))
        stack_inp = torch.cat([alt_comb, stack_inp], 1)
        stack_inp = self.bn0(stack_inp)
        x = self.inp_drop(stack_inp)
        res = x
        if self.way == 's':
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x + res
        else:
            conv1 = self.conv1(x).view(-1, self.p.channel, 400)
            conv2 = self.conv2(x).view(-1, self.p.channel, 400)
            conv3 = self.conv3(x).view(-1, self.p.channel, 400)
            res = res.expand(-1, self.p.channel, 20, 20).view(-1, self.p.channel, 400)
            x = torch.cat((res, conv1, conv2, conv3), dim=2)
            x = self.W_gate_e(x).view(-1, self.p.channel, 20, 20)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        if strategy == 'one_to_n':
            x = torch.mm(x, self.ent_embed.weight.transpose(1, 0))
            x += self.bias.expand_as(x)
        else:
            x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
            x += self.bias[neg_ents]

        pred = torch.sigmoid(x)

        return pred
