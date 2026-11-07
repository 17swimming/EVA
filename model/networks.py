import logging
from collections import OrderedDict
import pickle
import torch
from model.utils import img2col, get_density
import re
# from configs import *


def compute_num_OPS(nn):
    """
    计算神经网络中所有操作的总运算次数（OPS）
    
    参数:
    nn: 神经网络层列表，包含各种类型的网络层（全连接层、卷积层、LIF神经元等）
    
    返回:
    total_ops: 总运算次数
    FC_ops: 全连接层运算次数
    LIF_ops: LIF神经元运算次数
    """
    # 初始化各种运算次数计数器
    total_ops = 0
    FC_ops = 0
    LIF_ops = 0
    
    # 遍历网络中的每一层
    for op in nn:
        # 如果是全连接层（Fully Connected Layer）
        if isinstance(op, FC):
            # 计算矩阵维度：M(批量大小*时间步数*序列长度) × K(输入维度) × N(输出维度)
            M = op.batch_size * op.time_steps * op.sequence_length
            K = op.input_dim
            N = op.output_dim
            # 获取激活张量的密度（非零元素比例）
            density = get_density(op.activation_tensor.sparse_map)
            #todo 累加运算次数：M×K×N×密度（利用稀疏性减少计算量）————是不是过于理想化了，没有任何冗余计算
            total_ops += int(M * K * N * density)
            FC_ops += int(M * K * N * density)
            
        # 如果是卷积层（Convolutional Layer）
        elif isinstance(op, Conv2D):
            # 将卷积层转换为等效的全连接层进行计算
            eq_op = conv2d_2_fc(op)
            M = eq_op.batch_size * eq_op.time_steps * eq_op.sequence_length
            K = eq_op.input_dim
            N = eq_op.output_dim
            # 获取等效全连接层激活张量的密度
            density = get_density(eq_op.activation_tensor.sparse_map)
            # 累加运算次数
            total_ops += int(M * K * N * density)
            FC_ops += int(M * K * N * density)
            
        # 如果是LIF神经元层（Leaky Integrate-and-Fire Neuron Layer）
        elif isinstance(op, LIFNeuron):
            # LIF神经元运算次数 = 批量大小 × 时间步数 × 神经元数量
            total_ops += op.batch_size * op.time_steps * op.num_neuron
            LIF_ops += op.batch_size * op.time_steps * op.num_neuron
            
        # 如果是层归一化层（Layer Normalization）
        elif isinstance(op, LayerNorm):
            # 层归一化运算次数 = 批量大小 × 时间步数 × 序列长度 × 维度 × 2
            total_ops += op.batch_size * op.time_steps * op.sequence_length * op.dim * 2
            
        # 如果是注意力层（Attention Layer）
        elif isinstance(op, Attention):
            # 根据不同的注意力类型计算运算次数
            if op.attention_type == 'spikingbert':
                # SpikingBERT注意力机制的运算次数计算
                total_ops += op.batch_size * op.time_steps * op.num_head * op.sequence_length * op.sequence_length * op.dim_per_head * 2
                total_ops += op.batch_size * op.time_steps * op.num_head * op.sequence_length * op.sequence_length * 2
            else:
                raise ValueError('未实现的注意力类型')
    
    # 返回总运算次数和各类运算次数
    return total_ops, FC_ops, LIF_ops

class Tensor:
    """
    张量类，用于表示神经网络中的数据（激活值、权重、输出等）
    
    在脉冲神经网络中，数据可以是脉冲（spike，二值化数据）或浮点数（fp8）
    """
    def __init__(self, shape, dtype, sparse=False):
        # 标识是否为激活张量
        self.is_activation = True
        # 是否为稀疏张量（大部分元素为零）
        self.sparse = True if dtype == 'spike' else sparse
        # 张量形状（维度大小）
        self.shape = shape
        # 数据类型：'spike'(脉冲) 或 'fp8'(8位浮点数)
        self.dtype = dtype
        # 数据位宽：脉冲为1位，fp8为8位
        self.nbits = 8 if dtype == 'fp8' else 1
        # 对于FC，sparse_map为conv2d的sparse_map被img2col后的结果
        self.sparse_map = None

    def get_size(self):
        """
        获取张量大小（元素总数 × 位宽）
        """
        return torch.tensor(self.shape).prod().item() * self.nbits

class FC:
    """
    全连接层（Fully Connected Layer）类
    
    全连接层执行矩阵乘法操作：输出 = 激活值 × 权重
    在脉冲神经网络中，激活值通常是脉冲（二值化数据）
    """
    def __init__(self, name, input_dim, output_dim, sequence_length, batch_size, time_steps):
        self.name = name
        # 输入维度（特征数）
        self.input_dim = input_dim
        # 输出维度（特征数）
        self.output_dim = output_dim
        # tag序列长度=输出高度 × 输出高度
        self.sequence_length = sequence_length
        # 批量大小（一次处理的数据样本数）
        self.batch_size = batch_size
        # 时间步数（脉冲神经网络特有的时间维度）
        self.time_steps = time_steps
        # 激活张量：输入数据，通常是脉冲数据
        self.activation_tensor = Tensor([batch_size, time_steps, sequence_length, input_dim], 'spike', sparse=True)
        # 权重张量：网络参数，通常是浮点数
        self.weight_tensor = Tensor([input_dim, output_dim], 'fp8', sparse=False)
        # 输出张量：全连接层计算结果
        self.output_tensor = Tensor([batch_size, time_steps, sequence_length, output_dim], 'fp8', sparse=False)

class Conv2D:
    """
    二维卷积层（2D Convolutional Layer）类
    
    卷积层在图像处理中广泛应用，通过卷积核提取局部特征
    """
    def __init__(self, name, input_H, input_channel, output_channel, kernel_size, padding, stride, batch_size, time_steps, img2col=True):
        # 层名称
        self.name = name
        # 输入图像高度（假设图像为正方形）
        self.input_H = input_H
        # 卷积核大小
        self.kernel_size = kernel_size
        # 输出图像高度，根据卷积公式计算
        self.output_H = (input_H - kernel_size + 2 * padding) // stride + 1
        # 批量大小
        self.batch_size = batch_size
        # 时间步数（脉冲神经网络特有）
        self.time_steps = time_steps
        # 输入通道数
        self.input_channel = input_channel
        # 输出通道数
        self.output_channel = output_channel
        # 填充大小
        self.padding = padding
        # 步长
        self.stride = stride
        # 激活张量：已经使用img2col技术将卷积转换为矩阵乘法
        self.activation_tensor = Tensor([batch_size, time_steps, self.output_H * self.output_H, input_channel * kernel_size * kernel_size], 'spike', sparse=True)
        # 权重张量：卷积核参数
        self.weight_tensor = Tensor([input_channel * kernel_size * kernel_size, output_channel], 'fp8', sparse=False)
        # 输出张量
        self.output_tensor = Tensor([batch_size, time_steps, self.output_H * self.output_H, output_channel], 'fp8', sparse=False)

class MaxPool2D:
    """
    二维最大池化层（2D Max Pooling Layer）类
    
    池化层用于降采样，减少数据维度和计算量，同时保留重要特征
    """
    def __init__(self, name, input_H, channel, kernel_size, padding, stride, batch_size, time_steps, img2col=True):
        # 层名称
        self.name = name
        # 输入图像高度
        self.input_H = input_H
        # 池化核大小
        self.kernel_size = kernel_size
        # 输出图像高度
        self.output_H = (input_H - kernel_size + 2 * padding) // stride + 1
        # 批量大小
        self.batch_size = batch_size
        # 时间步数
        self.time_steps = time_steps
        # 通道数
        self.channel = channel
        # 激活张量
        self.activation_tensor = Tensor([batch_size, time_steps, self.output_H * self.output_H, channel, kernel_size * kernel_size], 'spike', sparse=True)
        # 输出张量
        self.output_tensor = Tensor([batch_size, time_steps, self.output_H * self.output_H, channel], 'spike', sparse=True)

class LIFNeuron:
    """
    LIF神经元（Leaky Integrate-and-Fire Neuron）类
    
    LIF是脉冲神经网络中最常用的神经元模型之一
    它模拟生物神经元的积分-发放机制
    """
    def __init__(self, name, num_neuron, batch_size, time_steps):
        # 层名称
        self.name = name
        # 神经元数量
        self.num_neuron = num_neuron
        # 批量大小
        self.batch_size = batch_size
        # 时间步数
        self.time_steps = time_steps
        # 激活张量：神经元输出的脉冲序列
        self.activation_tensor = Tensor([batch_size, time_steps, num_neuron], 'fp8', sparse=False)
        # 膜电位张量：神经元的内部状态
        self.membrane_potential = Tensor([batch_size, time_steps, num_neuron], 'fp8', sparse=False)
        # 输出张量：最终输出的脉冲
        self.output_tensor = Tensor([batch_size, time_steps, num_neuron], 'spike', sparse=True)

class Attention:
    """
    注意力机制层类
    
    注意力机制允许模型关注输入的不同部分，是Transformer架构的核心组件
    """
    def __init__(self, name, dim, sequence_length, num_head, batch_size, time_steps, attention_type='spikformer'):
        # 层名称
        self.name = name
        # 模型维度
        self.dim = dim
        # 序列长度
        self.sequence_length = sequence_length
        # 注意力头数
        self.num_head = num_head
        # 每个注意力头的维度
        self.dim_per_head = dim // num_head
        # 批量大小
        self.batch_size = batch_size
        # 时间步数
        self.time_steps = time_steps
        # 注意力类型
        self.attention_type = attention_type
        # 查询（Query）激活张量
        self.act_q_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        # 键（Key）激活张量
        self.act_k_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        # 值（Value）激活张量
        self.act_v_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        
        # 根据不同的注意力类型初始化张量
        if attention_type == 'spikformer' or attention_type == 'spikebert':
            # 注意力权重张量
            self.attn_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, sequence_length], 'fp8', sparse=True)
            # 输出张量
            self.output_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'fp8', sparse=False)
        elif attention_type == 'sdt':
            # 键值输出张量
            self.kv_out_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
            # 输出张量
            self.output_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        elif attention_type == 'spikingbert':
            # 注意力权重张量
            self.attn_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, sequence_length], 'fp8', sparse=False)
            # 输出张量
            self.output_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'fp8', sparse=False)
        else:
            raise ValueError('未知的注意力类型')

class LayerNorm:
    """
    层归一化（Layer Normalization）层类
    
    层归一化对每个样本的所有特征进行归一化，有助于训练稳定
    """
    def __init__(self, name, dim, batch_size, sequence_length, time_steps):
        # 层名称
        self.name = name
        # 维度
        self.dim = dim
        # 批量大小
        self.batch_size = batch_size
        # 序列长度
        self.sequence_length = sequence_length
        # 时间步数
        self.time_steps = time_steps
        # 激活张量
        self.activation_tensor = Tensor([batch_size, time_steps, sequence_length, dim], 'fp8', sparse=False)
        # 输出张量
        self.output_tensor = Tensor([batch_size, time_steps, sequence_length, dim], 'fp8', sparse=False)     


def conv2d_2_fc(operator: Conv2D) -> FC:
    """
    将卷积层转换为等效的全连接层
    tag 每个卷积核对应一个全连接层
    通过img2col技术将卷积操作转换为矩阵乘法操作
    """
    # 计算等效全连接层的输入维度：卷积核大小 × 卷积核大小 × 输入通道数
    eq_input_dim  = operator.kernel_size * operator.kernel_size * operator.input_channel
    #tag 等效全连接层的序列长度：输出高度 × 输出高度
    eq_sequence_length = operator.output_H * operator.output_H
    # 输出维度等于卷积层的输出通道数
    eq_output_dim = operator.output_channel
    # 使用img2col将卷积input tensor转换为矩阵乘法的输入（M，k*k*Cin）
    eq_sparse_map = img2col(operator.activation_tensor.sparse_map, operator.kernel_size, operator.stride, operator.padding)
    # tag 创建等效的全连接层
    # (name, input_dim, output_dim, sequence_length, batch_size, time_steps):
    eq_fc = FC(operator.name + '_kernel' + str(operator.kernel_size) + '_fc', eq_input_dim, eq_output_dim, eq_sequence_length, operator.batch_size, operator.time_steps)
    # 设置等效全连接层的激活张量
    eq_fc.activation_tensor.sparse_map = eq_sparse_map
    return eq_fc

def extract_kernel_size(s):
    """
    从字符串中提取卷积核大小
    
    参数:
    s: 包含卷积核信息的字符串
    
    返回:
    卷积核大小（整数）
    """
    # 使用正则表达式匹配'kernel'和'_fc'之间的数字
    match = re.search(r'kernel(.*?)_fc', s)
    if match:
        return int(match.group(1))
    return 1 # 假设是全连接层

def create_network(name, spike_info):
    """
    创建指定名称的神经网络
    
    参数:
    name: 网络名称
    spike_info: 包含脉冲信息的文件路径
    
    返回:
    神经网络层列表
    """
    # 从文件名中提取数据集名称
    dataset = spike_info.split('_')[1].split('.')[0]
    # 转换为小写
    name = name.lower()
    
    # 根据网络名称加载相应的配置
    if name == 'spikformer':
        config = spikformer_config(dataset=dataset)
    elif name == 'sdt':
        config = SDT_config(dataset=dataset)
    elif name == 'vgg16':
        config = vgg16_config()
    elif name == 'vgg9':
        config = vgg9_config(dataset=dataset)
    elif name == 'lenet5':
        config = lenet5_config()
    elif name == 'resnet18':
        config = resnet18_config()
    elif name == 'spikebert':
        config = spikeBERT_config(dataset=dataset)
    elif name == 'spikingbert':
        config = spikingBERT_config(dataset=dataset)
    else:
        raise ValueError('未知的网络名称')
    
    # 初始化网络层列表
    ops = []
    # 根据配置创建网络层
    for key, value in config.items():
        if key.startswith('conv2d'):
            ops.append(Conv2D(key, *value))
        elif key.startswith('lif'):
            ops.append(LIFNeuron(key, *value))
        elif key.startswith('maxpool2d'):
            ops.append(MaxPool2D(key, *value))
        elif key.startswith('fc'):
            ops.append(FC(key, *value))
        elif key.startswith('attention'):
            ops.append(Attention(key, *value, attention_type=name))
        elif key.startswith('layernorm'):
            ops.append(LayerNorm(key, *value))

    # 加载脉冲信息并设置到对应的网络层
    with open(spike_info, 'rb') as f:
        sparse_act = pickle.load(f)
        for op in ops:
            if isinstance(op, Attention):
                # 为注意力层设置查询、键、值的脉冲信息
                op.act_k_tensor.sparse_map = sparse_act[op.name + '_k'].contiguous()
                op.act_v_tensor.sparse_map = sparse_act[op.name + '_v'].contiguous()
                if name != 'spikingbert':
                    op.act_q_tensor.sparse_map = sparse_act[op.name + '_q'].contiguous()
            elif isinstance(op, Conv2D) or isinstance(op, FC):
                # 为卷积层或全连接层设置激活脉冲信息
                if op.name in sparse_act:
                    op.activation_tensor.sparse_map = sparse_act[op.name].contiguous()
                else:
                    raise ValueError(f'{op.name} 在 {spike_info} 中未找到')
    return ops
    
def print_sparsity(network, spike_info):
    """
    打印网络中各层的稀疏度信息
    
    参数:
    network: 网络名称
    spike_info: 包含脉冲信息的文件路径
    """
    if network == 'spikformer':
        with open(spike_info, 'rb') as f:
            sparse_act = pickle.load(f)
            for key, value in sparse_act.items():
                logging.info(f'{key}: {get_density(value):.4f}')
    elif network == 'spikebert':
        with open(spike_info, 'rb') as f:
            sparse_act = pickle.load(f)
            for key, value in sparse_act.items():
                logging.info(f'{key}: {get_density(value):.4f}')
    elif network == 'sdt':
        with open(spike_info, 'rb') as f:
            sparse_act = torch.load(f)
            for key, value in sparse_act.items():
                logging.info(f'{key}: {get_density(value):.4f}')
    elif network == 'vgg16':
        with open(spike_info, 'rb') as f:
            sparse_act = pickle.load(f)
            for key, value in sparse_act.items():
                logging.info(f'{key}: {get_density(value):.4f}')

if __name__ == '__main__':
    """
    主函数，用于测试网络创建和运算次数计算
    """
    logging.basicConfig(level=logging.INFO)
    # 创建spikingbert网络
    nn = create_network('spikingbert', 'data/spikingbert_sst2.pkl')
    # 计算网络中的运算次数
    ops, fc_ops, lif_ops = compute_num_OPS(nn)
    print("总运算次数: ", ops)
    print("全连接层运算次数: ", fc_ops)
    print("LIF神经元运算次数: ", lif_ops)
    # print_sparsity('spikeBERT', 'data/vgg16_cifar100.pkl')

    # total_ops = compute_num_OPS(ops)
    # num_adder = 128
    # frequency = 500 * 1024 * 1024
    # accelerator_ops = num_adder * frequency
    # accelerator_time = total_ops / accelerator_ops
    # logging.info(f'总运算次数: {total_ops}')
    # logging.info(f'加速器时间: {accelerator_time} s')
    # print_sparsity('vgg16', 'vgg16_cifar10_new.pkl')
    # attention 32 * 64, 64 * 32, 12 32 * 32 * 2 * 12


