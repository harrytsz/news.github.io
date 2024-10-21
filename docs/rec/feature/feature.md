## 特征工程部分

数据探索是解决实际场景问题最基本的步骤，一个问题能不能快速找到有效方案，在很大程度上依赖于数据探索。下面从缺失值处理、样本不平衡、常见的数据分布三个方面探讨数据探索过程。

### 缺失数据处理

#### 发现缺失数据
可以使用 Pandas 中的 count() 函数以及 shape() 函数进行统计：count()函数可以统计不为空数据的个数；shape()函数可以统计数据样本的个数；将 shape() 函数减去 count() 函数就可以得到数据的缺失个数，再用缺失个数除以样本的个数来计算样本中此字段的缺失率。

#### 缺失值处理
- dropna 根据各标签中的值是否存在缺失数据对轴标签进行过滤，可通过阈值调节对缺失值的容忍度
- fillna 用指定值或插值方法填充缺失值
- isnull 返回一个含有 bool 型的对象，这些 bool 型的值表示哪些是缺失值 NaN
- notnull isnull 的否定式

针对缺失值较少的情况，一般采用下面几种处理方法：
- 统计量填充
    - 连续值：推荐使用中位数，排除一些特别大或者特别小的异常值造成的影响
    - 离散值：推荐使用众数，不能用均值和中位数
- 特殊值处理： 填一个不在这个正常取值范围内的值，如使用-999, 0 等表示缺失值
- 不处理：XGBoost和 LightGBM这两种比赛常用的预测模型对缺失值并不敏感，算法本身有一套缺失值处理算法

### 不均衡样本
在实际场景中有很多数据存在不均衡的现象，如“长尾现象”，也就是我们常说的“二八原理”。在分类任务中，不同类别的训练样本数目常存在差异很大的情况，这时样本不均衡会出现模型对样本数较多的类别过拟合，对较少的类别欠拟合的现象，即总是将样本分到样本数较多的类别中。

#### 不均衡样本处理
- 随机欠采样
- 随机过采样
- 基于聚类的过采样方法
- SMOTE 算法
- 基于数据清洗的 SMOTE

### 常见的数据分布
概率分布用于表述随机变量取值的概率规律，其实所有事件都可以认为是随机变量，故都可以用概率表示。
离散型概率分布也称为：概率质量函数，常见的有：伯努利分布、二项分布、泊松分布和几何分布等。
连续型概率分布也称为：概率密度函数，常见的有：正态分布、指数分布、ß分布等。

### 特征归一化
特征归一化的目的是消除数据特征之间的量纲影响，使不同指标之间具有可比性。
- MaxMin Normalization
$$x_{norm} = \frac{X-X_{min}}{X_{max}-X_{min}}$$

- Z-Score Normalization
$$z = \frac{x-\mu}{\sigma}$$

通过梯度下降求解的模型，归一化会对收敛速度产生影响，因此在通常情况下都需要归一化处理。这些模型包括线性回归、逻辑回归、支持向量机、神经网络等。**但对于决策树模型一般不需要进行归一化处理**。

### 类别型特征的转换
类别型特征的常用处理方法：
- 序号编码
- One-Hot 编码
- 二进制编码

### 文本表示模型
#### 词袋模型
词袋模型是最基础的文本表示模型，是将文章以词为单位切分开（将每篇文章看成一袋子的单词，并忽略每个单词出现的顺序），将文章表示成一个长向量，每一维代表一个单词，该维的权重表示重要程度，常用 TF-IDF 来计算权重。
$$TF-IDF(t,d) = TF(t,d) * IDF(t)$$
其中， TF(t, d) 为单词 t 在文档 d 中出现的频率，IDF(t)是逆文档频率，用来衡量单词 t 对表达语义所起的重要性，表示为:
$$IDF(t)=log\frac{a}{b+1}$$
其中，a = 文章总数，b = 包含单词 t 的文章总数.

#### N-gram 模型
N-gram 模型是一种语言模型，对于由 N 个词组成的语句片段，我们假设第 N 个词出现与否只与前 N-1个词相关，整个语句出现的概率就是这个 N 个词概率的乘积。

#### 主题模型
词袋模型和 N-gram 模型都无法识别两个不同的词或词组是否具有相同的主题，而主题模型可以将具有相同主题的词或词组映射到同一维度上，映射到的这个维度表示某个主题。

#### 词嵌入
词嵌入是一类将词向量化的模型的统称，核心思想是将每个词都映射到低维空间上的一个稠密向量。简单来说，嵌入是用向量表示一个物体，这个物体可以是一个单词、一条语句、一个序列、一件商品、一个动作、一本书、一部电影等。可以说嵌入涉及机器学习、深度学习的绝大部分对象，这些对象是机器学习和深度学习中最基本、最常用、最重要的对象。
尤其 word2vec 这样的 Word Embedding 的广泛应用，更是带来了更大范围的延伸和扩展，嵌入技术由最初的自然语言处理领域向传统机器学习、搜索排序、推荐、知识图谱等领域延伸，具体表现为由 Word Embedding 向 Item Embedding、Graph Embedding、Categorical variables Embedding 等方向延伸。
Embedding 本身也在不断更新，由最初表现单一的静态向表现更丰富的动态延伸和拓展。具体表现为静态的 Word Embedding 向 ELMo、Transformer、GPT、BERT、XLNet、ALBERT 等动态的预训练模型延伸。

#### 如何处理序列问题

序列问题非常常见，如自然语言处理、网页浏览、时间序列等都与序列密不可分。

拿到一份语言材料后，不管是中文还是英文的，首先需要做一些必要的清理工作，如清理特殊符号、格式转换、过滤停用词等。然后进行分词、索引化，再利用相关模型或算法把单词、词等标识符向量化，最后输出给下游任务。

![](https://pic.imgdb.cn/item/6141d9ff2ab3f51d914ce590.png)

其中，词嵌入或预训练模型是关键，他们的质量好坏直接影响下游任务的效果。

#### Word Embedding

One-Hot Encoding 这种编码方法虽然方便简洁，但非常稀疏，属于硬编码，而且无法承载更多信息。后来，人们采用数值向量或标识符嵌入 Token Embedding 来表示，也称为 Word Embedding 或分布式表示。One-hot Encoding 是稀疏、高维的硬编码，如果一个语料有一万个不同的词，那么每个词就需要用一万维独热编码表示。如果用向量或词嵌入表示，那么这些向量就是低维、密集的，而且这些向量值都是通过学习获得的，而不是硬性设定的。因此，Embedding 可以纳入更多的语义层面的信息。

词嵌入向量 Embedding 的学习方法，通常有两种：
- 利用算法模型的 Embedding 层学习词嵌入
    > 把 Embedding 作为第一层，先随机初始化这些词向量，然后利用算法框架（Pytorch、Tensorflow)不断学习，包括正向传播以及反向传播，最后得到需要的词向量。

- 使用预训练的词嵌入
    > 利用在较大语料上预训练好的词嵌入或预训练模型，把这些词嵌入加载到当前任务或模型中。预训练模型很多：Word2Vec、ELMo、Bert、XLNet、ALBert等。
    
#### Word2Vec 实现
**Pytorch版本**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim
from torch.nn.parameter import Parameter
from collections import Counter
import numpy as np
import random
import math
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA = torch.cuda.is_available()

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(1224)
np.random.seed(1224)
torch.manual_seed(1224)

if USE_CUDA:
    torch.cuda.manual_seed(1224)

# 设定hyper parameters
C = 3  # context window, namely nearby words threshold
K = 100  # number of negative samples, 每出现一个正确的词就要出现100个错误的词
NUM_EPOCHS = 1
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100
LOG_FILE = 'word_embedding.log'

# Preprocessing
def word_tokenize(text):
    return text.split()

with open(file='./text8/text8.train.txt', mode='r') as fin:
    text = fin.read()

# 构建词汇表
text = [word for word in word_tokenize(text=text.lower())]
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)
word_freqs = word_freqs / np.sum(word_freqs)  # 重新normalized一次
VOCAB_SIZE = len(idx_to_word)

"""
为了使用Dataloader，我们需要定义以下两个function:
    - __len__(), 需要返回整个数据集中有多少item
    - __getitem__(), 根据给定的index返回一个item
有了Dataloader后，可以轻松随机打乱整个数据集，拿到一个batch的数据等。
"""
# DataLoader
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(word, word_to_idx['<unk>']) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):  # 返回数据用于训练
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], replacement=True)
        return center_word, pos_words, neg_words

# 创建dataset和dataloader
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataLoader = tud.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# 定义PyTorch模型
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        init_range = 0.5 / self.embed_size
        self.in_embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-init_range, init_range)
        self.out_embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size)
        self.out_embed.weight.data.uniform_(-init_range, init_range)

    def forward(self, input_labels, pos_labels, neg_lables):  # loss function
        """
        :param input_labels: [batch_size]
        :param pos_labels: [batch_size, (window_size * 2)]
        :param neg_lables: [batch_size, (window_size * 2 * K)]
        :return: loss, [batch_size]
        """
        batch_size = input_labels.size(0)

        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window_size * 2), embed_size]
        neg_embedding = self.out_embed(neg_lables)  # [batch_size, (window_size * 2 * K), embed_size]
        # unsqueeze()升维, squeeze()降维
        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1], 第二个维度加1
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze()  # [batch_size, (window_size * 2)]
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze()  # [batch_size, (window_size * 2 * K)]
        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)
        loss = log_pos + log_neg
        return -loss

    def input_embedding(self):  # 取出self.in_embed数据参数
        return self.in_embed.weight.data.cpu().numpy()

# 定义一个模型以及把模型移动到GPU
model = EmbeddingModel(vocab_size=VOCAB_SIZE, embed_size=EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()

# 评估模型
def evaluate(filename, embedding_weights):
    if filename.endswith('.csv'):
        data = pd.read_csv(filename, sep=',')
    else:
        data = pd.read_csv(filename, sep='\t')
    human_similarity = []
    model_similarity = []
    for i in data.iloc[:, 0:2].index:
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            model_similarity.append(float(cosine_similarity(word1_embed, word2_embed)))
            human_similarity.append(float(data.iloc[i, 2]))
    return scipy.stats.spearmanr(human_similarity, model_similarity)

def find_nearest(word):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]

# 训练模型
"""
1.模型一般需要训练若干个epoch，每个epoch我们都把所有数据分成若干个batch，把每个batch的输入和输出都包装成cuda tensor；
2.forward pass，通过输入的句子预测每个单词的下一个单词，用模型的预测和正确的下一个单词计算cross entropy loss；
3.清空模型当前的Gradient；
4.backward pass，更新模型参数；
5.每隔一定的iteration，输出模型在当前iteration的loss以及在验证数据集上做模型的评估。
"""
optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataLoader):
        # print(input_labels, pos_labels, neg_labels)
        # if i > 2:
        #     break
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()  # 传入参数给forward()函数
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                with open(file=LOG_FILE, mode='a', encoding='UTF-8') as f_out:
                    f_out.write('Epoch: {}, Iteration: {}, Loss: {} + \n'.format(e, i, loss.item()))
                    print(f'Epoch: {e}, Iteration: {i}, Loss: {loss.item()}')

            if i % 2000 == 0:
                embedding_weights = model.input_embedding()
                sim_simlex = evaluate(filename='simlex-999.txt', embedding_weights=embedding_weights)
                sim_men = evaluate(filename='men.txt', embedding_weights=embedding_weights)
                sim_353 = evaluate(filename='wordsim353.csv', embedding_weights=embedding_weights)
                with open(file=LOG_FILE, mode='a') as f_out:
                    print(f'Epoch: {e}, Iteration: {i}, sim_simlex: {sim_simlex}, sim_men: {sim_men}, sim_353: {sim_353}, nearest to monster: {find_nearest(word="monster")} + \n')
                    f_out.write('Epoch: {}, Iteration: {}, sim_simlex: {}, sim_men: {}, sim_353: {}, nearest to monster: {} + \n'.format(
                        e, i, sim_simlex, sim_men, sim_353, find_nearest(word="monster")))

    embedding_weights = model.input_embedding()
    np.save('embedding-{}'.format(EMBEDDING_SIZE), embedding_weights)
    torch.save(model.state_dict(), 'embedding-{}.th'.format(EMBEDDING_SIZE))
```

**Tensorflow 版本**

```python
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
 
'''######## Word2Vec Skip-Gram ##############'''
'''下载文本数据的函数'''
def maybe_download(filename, expected_bytes):
 
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)  #下载数据的压缩文件
    #核对文件尺寸
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verfied', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with browser?'
        )
    return filename

filename = maybe_download('text8.zip', 31344016)    #根据文件名和byte下载数据
 
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()    #解压文件并将数据转成单词的列表
    return data
 
words = read_data(filename)
print('Data size', len(words))
# print(words[:5]) 
vocabulary_size = 50000
 
'''创建vocabulary词汇表'''
def build_dataset(words):
    count = [['UNK', -1]]   #词频统计，['UNK', -1]表示未知单词的频次设为-1
    #使用collections.Counter()方法统计单词列表中单词的频数， 使用.most_common()方法取top50000频数的单词(频次从大到小)作为vocabulary
    #     # >> > Counter('abcdeabcdabcaba').most_common(3)
    #     # [('a', 5), ('b', 4), ('c', 3)]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    #创建一个词典dict，将top50000的单词存入，以便快速查询
    dictionary = dict()
    #将top50000单词按频次给定其编号存入dictionary，key为词，value为编号(从1到50000)
    for word, _ in count:   #_用作一个名字，来表示某个变量是临时的或无关紧要的
        dictionary[word] = len(dictionary)
        #print(len(dictionary))
 
    data = list()   #单词列表转换后的编码
    unk_count = 0
 
    for word in words:  #遍历单词列表
        if word in dictionary:  #如果该单词存在于dictionary中
            index = dictionary[word]    #取该单词的频次为编号
        else:   #如果dictionary中没有该单词
            index = 0   #编号设为0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count #未知单词数(除了top50000以外的单词
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #返回单词列表中单词转换为编码(编码为该单词的频次)的data列表、每个单词的频数统计count、词汇表dictionary及其反转形式reverse_dictionary
    return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words   #删除原始单词表，节约内存
print("Most common words (+UNK)", count[:5])    #打印未知单词和最高频单词及它们的频次
print("Sample data", data[:10], [reverse_dictionary[i] for i in data[:10]])  #打印单词列表中前10个单词的编号及单词本身
 
data_index = 0
 
'''生成word2vec的训练样本，返回目标单词编号batch数组和其对应语境编号labels数组
skip-gram模式将原始数据"the quick brown fox jumped"转为(quick,the),(qucik,brown),(brown,quick),(brown,fox)等样本'''
def generate_batch(batch_size,  #一个批次的大小
                   num_skips,   #num_skips为对每个单词生成多少个样本
                   skip_window): #指单词最远可以联系的距离，设为1代表只能跟紧邻的两个单词生成样本，比如quick只能和前后的单词生成(quick,the),(qucik,brown)
    global data_index   #单词序号设为global，确保在调用该函数时，该变量可以被修改
    #python 中assert断言是声明其布尔值必须为真的判定，其返回值为假，就会触发异常
    assert batch_size % num_skips == 0  #skip-gram中参数的要求
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)   #初始化batch,存放目标单词
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  #初始化labels，存放目标单词的语境单词们
    span = 2 * skip_window + 1  #对某个单词创建相关样本时会用到的单词数量，包含目标单词和它前后的单词
    buffer = collections.deque(maxlen=span) #创建一个最大容量为span的deque，即双向队列
 
    for _ in range(span):
        buffer.append(data[data_index]) #从序号data_index开始， 把span个单词的编码顺序读入buffer作为初始值，循环完后，buffer填满，里面为目标单词和需要的单词
        data_index = (data_index + 1) % len(data)
 
    for i in range(batch_size // num_skips):    #批次的大小➗每个单词生成的样本数=该批次中单词的数量
        target = skip_window    #即buffer中下标为skip_window的变量为目标单词
        targets_to_avoid = [skip_window]    #避免使用的单词列表，要预测的时语境，不包含单词本身
        for j in range(num_skips):
            while target in targets_to_avoid:   #生成随机数直到不在targets_to_avoid中，代表可以使用的语境单词
                target = random.randint(0, span-1)
            targets_to_avoid.append(target)     #该语境单词已使用，加入避免使用的单词列表
            batch[i * num_skips + j] = buffer[skip_window]  #feature即目标词汇
            labels[i * num_skips + j, 0] = buffer[target]   #label即当前语境单词
        buffer.append(data[data_index]) #对一个目标单词生成完所有样本后，再读入下一个单词(同时会抛掉buffer中第一个单词)
        data_index = (data_index + 1) % len(data)   #单词序号+1
    #获得了batch_size个训练样本，返回目标单词编号batch数组和其对应语境单词编号labels数组
    return batch, labels
 
'''测试word2vec训练样本生成'''

batch_size = 128
embedding_size = 128    #单词转为词向量的维度，一般为50-1000这个范围内的值
skip_window = 1
num_skips = 2
 
#生成验证数据valid_samples
valid_size = 16     #用来抽取的验证单词数
valid_window = 100  #验证单词从频数最高的100个单词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  #从valid_window中随机抽取valid_size个数字,返回一维数组
num_sampled = 64    #训练时用来做负样本的噪声单词的数量
 
'''定义Skip-Gram Word2Vec模型的网络结构'''
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)     #验证单词的索引   shape(1, 16)
 
    with tf.device('/cpu:0'):   #限定所有计算在CPU上执行，因为接下去的一些计算操作在GPU上可能还没有实现
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))   #随机生成所有单词的词向量embeddings，范围[-1, 1]
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)    #在embeddings tensor中查找输入train_inputs编号对应的向量embed
        nce_weights = tf.Variable(  #使用tf.truncated_normal截断的随机正态分布初始化NCE Loss中的权重参数nce_weights
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))   #偏置初始化为0
    #使用tf.nn.nce_loss计算出词向量embedding在训练数据上的loss，并使用tf.reduce_mean进行汇总
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,   #权重
                                         biases=nce_biases,     #偏置
                                         labels=train_labels,   #标记
                                         inputs=embed,          #
                                         num_sampled=num_sampled,   #负样本噪声单词数量
                                         num_classes=vocabulary_size))  #可能的分类的数量(单词数)
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
 
    '''余弦相似度计算'''
    #L2范数又叫“岭回归”，作用是改善过拟合，L2范数计算方法：向量中各元素的平方和然后开根号
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)) #计算嵌入向量embeddings的L2范数
    normalized_embeddings = embeddings / norm   #embeddings除以其L2范数得到标准化后的normalized_embeddings
    valid_embeddings = tf.nn.embedding_lookup(  #根据验证单词的索引valid_dataset，查询验证单词的嵌入向量
        normalized_embeddings, valid_dataset)
    # 计算验证单词的嵌入向量与词汇表中所有单词的相似性, valid_embeddings * (normalized_embeddings的转置)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()
 
'''可视化Word2Vec，low_dim_embds为降维到2维的单词的空间向量'''
def plot_with_labels(low_dim_embds, labels, filename='tsne.png'):
    assert low_dim_embds.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))    #图片大小
 
    for i, label in enumerate(labels):
        x, y = low_dim_embds[i,:]
        plt.scatter(x, y)   #显示散点图
        plt.annotate(label,     #展示单词本身
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)
 
'''测试'''
num_steps = 100001  #训练100001轮
 
with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")
    average_loss = 0
    for step in range(num_steps):
        #获得训练数据和数据标记
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)   #执行优化器运算和损失计算
        average_loss += loss_val    #损失值累加
        if step % 2000 == 0:    #每两千轮计算几次平均损失值
            if step > 0:
                average_loss /= 2000
            print("Average loss at step", step, ": ", average_loss)
            average_loss = 0
        #每一万轮，计算一次验证单词与全部单词的相似度，并将与每个验证单词最相近的8各单词展示出来
        if step % 10000 == 0:
            # tensor.eval():在`Session`中评估这个张量。调用此方法将执行所有先前的操作，这些操作将生成生成此张量的操作所需的输入。
            sim = similarity.eval()     #shape,(16, 50000)
            #print(tf.shape(sim).eval())
            for i in range(valid_size): #对每一个验证单词
                valid_word = reverse_dictionary[valid_examples[i]]  #根据前面随机抽取的验证单词编号(即频次)，在反转字典中取出该验证单词
                top_k = 8
                #.argsort()从小到大排列，返回其对应的索引，由于-sim()，所以返回的索引是相似度从大到小的
                nearest = (-sim[i, :]).argsort()[1:top_k+1]     #计算得到第i个验证单词相似度最接近的前8个单词的索引
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]     #相似度最接近的第i个单词
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval() #最终训练完的词向量矩阵
 
'''展示降维后的可视化效果'''
#使用sklearn.manifold.TSNE实现降维，这里直接将原始的128维嵌入向量降到2维
tsne = TSNE(perplexity=30,      #困惑度，默认30
            n_components=2,     #降到多少维
            init='pca',         #初始化的嵌入
            n_iter=5000)        #优化的最大迭代次数。至少应该是250。
plot_only = 150
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])  #进行降维，输入shape为 (n_samples, n_features) or (n_samples, n_samples)
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)  #用该可视化函数进行展示
```

#### Graph Embedding
Graph Embedding 和 Word Embedding 一样，目的是用低维、稠密、实值的向量表示网络中的节点。目前 Graph Embedding 在推荐系统、搜索排序、广告等领域非常流行，并且也取得了非常好的效果。Graph 表示二维关系，而 Sequence 表示一维关系，因此要将 Graph 转换为 Graph Embedding，一般需要先通过算法把 Graph 变为 Sequence，然后通过模型或算法把这些 Sequence 转换为 Embedding。
![](https://pic.imgdb.cn/item/6142cb782ab3f51d91d89e49.jpg)

##### DeepWalk 方法
DeepWalk 方法首先以随机游走(Random Walk) 的方式在网络中进行节点采样，生成序列数据，然后使用 Skip-gram 模型将序列转换为 Embedding。RandomWalk 是一种可重复访问已经访问过的节点的深度优先遍历算法。给定当前访问的起始点，从其邻居节点中随机选择一个节点作为下一个访问节点，重复此过程，直到访问序列长度满足预设值为止。

#### LINE 方法
DeepWalk 只适用无向、无权图。在2015年，微软亚研院发布了 LINE(Large-Scale Information Network Embedding)大型信息网络嵌入。LINE 方法使用边采样方法克服了传统的随机梯度法容易出现的 Node Embedding 聚集问题，同时提高了最后结果的效率和效果。
LINE 方法可以应用于有向图、无向图以及有权图，并且能够通过将一阶、二阶的邻近关系引入目标函数，使得最终学习到的 Node Embedding 的分布更为均衡、平滑。

#### Node2Vec 方法
在 DeepWalk 和 LINE 方法的基础之上，斯坦福大学在 2016 年发布了 Node2Vec，该算法不但关注了同质性和结构性，还可以在两者之间进行权衡。
同质性相同的物品很可能是同品类、同属性或者经常被一同购买的物品，而结构性相同的物品则是各个品类中的爆款、最佳凑单商品等拥有类似趋势的物品。同质性和结构性在推荐系统中都有着重要的作用，Node2Vec 这种特性可以用来发掘不同特征信息。

##### Embedding 

Item2Vec 是基于自然语言处理模型 Word2Vec 提出的，所以 Item2Vec 要处理的是类似文本句子、阅读序列之类的序列型数据。下面利用用户浏览新闻的日志数据训练 Item2Vec，首先需要按照时间戳排序，再 groupBy user_id 操作聚合每个用户的浏览记录。

```sql
+--------+-----------+--------+----------------------------+
|user_id | user_name | new_id |     new_browse_time        |
+--------+-----------+--------+----------------------------+
|  1296  |    张三   | 200000 | 2020-10-23 07:38:45.000000 |
|  1296  |    张三   | 300000 | 2020-10-23 07:38:49.000000 |
|  1296  |    张三   | 400002 | 2020-10-23 07:38:52.000000 |
|  1296  |    张三   | 700001 | 2020-10-23 07:38:55.000000 |
|  1296  |    张三   | 400000 | 2020-10-23 07:39:00.000000 |
|  1296  |    张三   | 200000 | 2020-10-23 07:39:04.000000 |
|  1296  |    张三   | 300001 | 2020-10-23 07:39:10.000000 |
|  1296  |    张三   | 700212 | 2020-10-23 07:40:41.000000 |
|  1296  |    张三   | 700219 | 2020-10-23 07:40:47.000000 |
|  1296  |    张三   | 200028 | 2020-10-23 07:41:09.000000 |
|  1296  |    张三   | 700255 | 2020-10-23 07:41:18.000000 |
|  1296  |    张三   | 700273 | 2020-10-23 07:41:29.000000 |
|  1296  |    张三   | 700311 | 2020-10-23 07:41:49.000000 |
|  1296  |    张三   | 700309 | 2020-10-23 07:41:56.000000 |
|  1296  |    张三   | 700306 | 2020-10-23 07:42:03.000000 |
|  1296  |    张三   | 700209 | 2020-10-23 07:42:10.000000 |
|  1296  |    张三   | 700224 | 2020-10-23 07:43:23.000000 |
|  1296  |    张三   | 700250 | 2020-10-23 07:44:16.000000 |
|  1296  |    张三   | 700273 | 2020-10-23 07:44:24.000000 |
|  1296  |    张三   | 700250 | 2021-07-13 17:18:01.000000 |
|  1296  |    张三   | 700219 | 2021-07-13 17:18:07.000000 |
|  1296  |    张三   | 700263 | 2021-07-13 17:18:15.000000 |
|  1296  |    张三   | 700224 | 2021-07-13 17:18:27.000000 |
|  1296  |    张三   | 700224 | 2021-07-13 17:19:49.000000 |
|  1296  |    张三   | 700311 | 2021-07-13 17:19:59.000000 |
|  1296  |    张三   | 700197 | 2021-07-13 17:20:11.000000 |
```
通过数据处理之后，得到新闻 new_id 组成的序列数据。准备好训练数据后，调用 Spark MLlib 中的 Word2Vec 模型接口。

首先创建 Word2Vec 模型并设定模型参数，Word2Vec 模型的关键参数有 3 个，分别是 setVectorSize、setWindowSize 以及 setNumIterations。其中， setVectorSize 用于设定生成的 Embedding 向量的维度，setWindowSize 用于设定在序列数据上采样的滑动窗口大小，setNumIterations 用于设定训练时的迭代次数。这些超参数的具体选择要根据实际的训练效果来做调整。模型的训练过程并不复杂，调用模型的 fit 接口就可以自动完成模型训练。训练完成后，模型会返回一个包含了所有模型参数的对象。
最后一步就是提取和保存生成的 Embedding 向量，直接调用 getVectors 接口就可以获得某个 new_id 对应的 Embedding 向量，之后就可以将这些 Embedding 向量保存到数据库或者缓存中。

```python
import os
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.mllib.feature import Word2Vec
from pyspark.ml.linalg import Vectors
import random
from collections import defaultdict
import numpy as np
from pyspark.sql import functions as F

class UdfFunction:
    @staticmethod
    def sortF(new_list, new_browse_time_list):
        """
        sort by time and return the corresponding news sequence
        eg:
            input: new_list:[1,2,3]
                   new_browse_time_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        for m, t in zip(new_list, new_browse_time_list):
            pairs.append((m, t))
        # sort by time
        pairs = sorted(pairs, key=lambda x: x[1])
        return [x[0] for x in pairs]

def processItemSequence(spark, rawSampleDataPath):
    # rating data
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)
    # ratingSamples.show(5)
    # ratingSamples.printSchema()
    sortUdf = udf(UdfFunction.sortF, ArrayType(StringType()))
    userSeq = ratingSamples \
        .groupBy("userId") \
        .agg(sortUdf(F.collect_list("new_id"), F.collect_list("new_browse_time").withColumn(d,checkpoint.cast(TimestampType()))).alias('new_ids')) \
        .withColumn("new_idStr", array_join(F.col("new_ids"), " "))
    return userSeq.select('new_idStr').rdd.map(lambda x: x[0].split(' '))

def trainItem2vec(spark, samples, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    word2vec = Word2Vec().setVectorSize(embLength).setWindowSize(5).setNumIterations(10)
    model = word2vec.fit(samples)
    synonyms = model.findSynonyms("158", 20)
    for synonym, cosineSimilarity in synonyms:
        print(synonym, cosineSimilarity)
    embOutputDir = '/'.join(embOutputPath.split('/')[:-1])
    if not os.path.exists(embOutputDir):
        os.makedirs(embOutputDir)
    with open(embOutputPath, 'w') as f:
        for new_id in model.getVectors():
            vectors = " ".join([str(emb) for emb in model.getVectors()[new_id]])
            f.write(new_id + ":" + vectors + "\n")
    embeddingLSH(spark, model.getVectors())
    return model

if __name__ == '__main__':
    conf = SparkConf().setAppName('ctrModel').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # Change to your own filepath
    file_path = '/src/main/resources'
    rawSampleDataPath = file_path + "/sampledata/newbrowse.csv"
    embLength = 10
    samples = processItemSequence(spark, rawSampleDataPath)
    model = trainItem2vec(spark, samples, embLength,
                          embOutputPath=file_path[7:] + "/modeldata2/item2vecEmb.csv", saveToRedis=False,
                          redisKeyPrefix="i2vEmb")
    graphEmb(samples, spark, embLength, embOutputFilename=file_path[7:] + "/modeldata2/itemGraphEmb.csv",
             saveToRedis=True, redisKeyPrefix="graphEmb")
    generateUserEmb(spark, rawSampleDataPath, model, embLength,
                    embOutputPath=file_path[7:] + "/modeldata2/userEmb.csv", saveToRedis=False,
                    redisKeyPrefix="uEmb")
```
模型训练完成后，验证一下训练的结果是不是合理。随机点击一篇新闻，通过 Item2Vec 得到的相似新闻列表如右侧推荐栏中展示的。从直观上可以判断一下这个推荐结果的合理性。
![](https://pic.imgdb.cn/item/6142b4c42ab3f51d91ba2cdb.jpg)
从图中可以看出，相似推荐的结果中有 3 篇新闻是相似的，另外 2 篇是不相似的。主要原因是本次系统中新闻的相似度计算仅仅更新了部分数据，所以得到的推荐结果不够准确。

#### Graph Embedding 
上面介绍的是利用 Item2Vec 算法生成新闻 Embedding 的过程。接下来基于 Deep Walk 方法实现一下 Graph Embedding 算法并生成新闻 Embedding。在 Deep Walk 方法中，首先需要准备的最关键数据是新闻之间的转移概率矩阵。转移概率矩阵表达了新闻关系图，它定义了随机游走的过程中，从新闻 A 跳转到新闻 B 的跳转概率。下面就是利用 Spark 生成的转移概率矩阵的过程：

```python
def generate_pair(x):
    pairSeq = []
    previousItem = ''
    for item in x:
        if not previousItem:
            previousItem = item
        else:
            pairSeq.append((previousItem, item))
            previousItem = item
    return pairSeq

def generateTransitionMatrix(samples):
    pairSamples = samples.flatMap(lambda x: generate_pair(x))
    pairCountMap = pairSamples.countByValue()
    pairTotalCount = 0
    transitionCountMatrix = defaultdict(dict)
    itemCountMap = defaultdict(int)
    for key, cnt in pairCountMap.items():
        key1, key2 = key
        transitionCountMatrix[key1][key2] = cnt
        itemCountMap[key1] += cnt
        pairTotalCount += cnt
    transitionMatrix = defaultdict(dict)
    itemDistribution = defaultdict(dict)
    for key1, transitionMap in transitionCountMatrix.items():
        for key2, cnt in transitionMap.items():
            transitionMatrix[key1][key2] = transitionCountMatrix[key1][key2] / itemCountMap[key1]
    for itemid, cnt in itemCountMap.items():
        itemDistribution[itemid] = cnt / pairTotalCount
    return transitionMatrix, itemDistribution
```
生成转移概率矩阵的函数的输入在训练 Item2Vec 的时候就已经处理好的用户新闻阅读序列数据。输出的是转移概率矩阵，由于转移概率矩阵比较稀疏，因此这里没有使用二维数组进行存储，而是采用二维 Map 数据结构存储。我们很容易就能获得从新闻 A 到新闻 B的转移概率，转移概率即为：transferMatrix(newA)(newB)。

#### 转移概率矩阵的计算原理
利用 pySpark 的 flatMap 操作把新闻阅读序列打散成一个个新闻对，再使用 countByValue 操作统计新闻对的数量，最后根据这些新闻对的数量求的每篇新闻之间的转移概率。

在计算出转移概率矩阵之后，我们就开始随机游走的采样过程。

#### Graph Embedding 随机游走采样

随机游走采样利用转移概率矩阵生成新的样本序列。首先，根据新闻出现次数的分布随机选择一篇作为起始，之后就开始随机游走的采样过程。在每次游走时，根据转移概率矩阵查找到两篇新闻之间的转移概率，然后根据这个概率进行跳转。比如，当前的新闻是A，从转移概率矩阵中查找到新闻 A 可能跳转到新闻 B 和新闻 C，它们的转移概率分别为：0.35 和 0.65 ，那么就按照这个概率值来随机选择新闻 B 或新闻 C。依次采样下去，直到收集足够的样本为止。
实现代码如下：

```python
def onceRandomWalk(transitionMatrix, itemDistribution, sampleLength):
    sample = []
    randomDouble = random.random()
    firstItem = ""
    accumulateProb = 0.0
    for item, prob in itemDistribution.items():
        accumulateProb += prob
        if accumulateProb >= randomDouble:
            firstItem = item
            break
    sample.append(firstItem)
    curElement = firstItem
    i = 1
    while i < sampleLength:
        if (curElement not in itemDistribution) or (curElement not in transitionMatrix):
            break
        probDistribution = transitionMatrix[curElement]
        randomDouble = random.random()
        accumulateProb = 0.0
        for item, prob in probDistribution.items():
            accumulateProb += prob
            if accumulateProb >= randomDouble:
                curElement = item
                break
        sample.append(curElement)
        i += 1
    return sample

def randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength):
    samples = []
    for i in range(sampleCount):
        samples.append(onceRandomWalk(transitionMatrix, itemDistribution, sampleLength))
    return samples

def graphEmb(samples, spark, embLength, embOutputFilename, saveToRedis, redisKeyPrefix):
    transitionMatrix, itemDistribution = generateTransitionMatrix(samples)
    sampleCount = 20000
    sampleLength = 10
    newSamples = randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength)
    rddSamples = spark.sparkContext.parallelize(newSamples)
    trainItem2vec(spark, rddSamples, embLength, embOutputFilename, saveToRedis, redisKeyPrefix)
```
