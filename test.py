"""
@FileName：test.py
@Description：
@Author：zhangyt\n
@Time：2024/12/2 15:18
"""
import torch


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
