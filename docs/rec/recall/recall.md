## 召回部分

从架构上看，召回层处于推荐系统的线上服务模块之中，推荐服务器从数据库或内存中获取所有候选物品集合后，会依次经过召回层、排序层、再排序层(补充策略算法层)才能生成用户最终看到的推荐列表。

为了提高计算速度，召回策略需要设计的尽可能简单，同时为了提高召回率、召回精度尽量把用户感兴趣的新闻都收集过来，这就要求召回策略不能过于简单，否则召回的新闻无法满足排序模型的要求。

### 单策略召回

单策略召回是指通过制定一条规则或者利用一个相对简单的模型算法来快速收集相关性高的物品。这里的规则可以是用户的兴趣偏好，比如用户最喜欢的新闻主题（经济、科技、体育等）、热度值最高的新闻等等。基于其中任何一条都可以构建一个单策略召回层，比如从用户 A 日志中统计得出用户对科技类新闻的阅读比例比较高，那么我们就从科技主题新闻中按照热度值召回 20 篇新闻放入排序候选集中。
单策略召回思路非常直观，召回速度非常快。“快”与“准”总是对立和统一的，这里的单策略召回有着自身的局限性。因为用户的兴趣是非常多元化的，如果一直给用户推荐某一种主题的新闻，用户的兴趣很快就会被消耗殆尽。所以在给用户推荐的时候，不仅要推用户曾经感兴趣的主题，还需要推荐给用户一些没有涉猎的主题，探索用户新的兴趣点。
这种动态的探索用户兴趣过程，显然单策略召回已经力不从心了。接下来介绍多路召回的方法

### 多路召回

多路召回是指采用不同的策略、特征或模型，分别召回一部分候选集，然后把候选集混合在一起供后续排序模型使用的策略。

多路召回是在计算速度和召回率之间进行权衡的折中，新闻推荐用到的多路召回策略，包括热门新闻、关键词召回、主题召回以及协同过滤召回等。除此之外，还可以使用集成算法思想，综合多种简单模型的推荐结果。

```python
def itemcf_sim(df, item_created_time_dict):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        思路: 基于物品的协同过滤 + 关联规则
    """
    user_item_time_dict = get_user_item_time(df)
    
    # 计算新闻相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于物品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if(i == j):
                    continue
                # 考虑文章的正向顺序点击和反向顺序点击    
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重，其中的参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重，其中的参数可以调节
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)      
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
    return i2i_sim_
```

在后续迭代版本中，考虑通过多线程并行、建立召回集缓存等手段进一步优化性能，同时在召回策略方面增加更多选择。

多路召回策略虽然能够比较全面地照顾到不同的召回方法，但也存在一些缺点。比如，在确定每一路的召回物品数量时，往往需要大量的人为干预，参数值需要经过大量的线上 AB 测试来确定。各种策略之间往往是割裂的，我们很难综合衡量不同策略对推荐的影响。

### 基于 Embedding 的召回方法
利用物品和用户 Embedding 向量的相似性构建召回层是业内非常经典的技术方案。多路召回中使用的兴趣偏好、热门度、流行趋势、物品属性等信息可以作为 Embedding 方法中的附加信息融合进最终的 Embedding 向量中。因此，在利用 Embedding 召回的过程中，就相当于考虑到了多路召回的多种策略。
另外，Embedding 向量召回的评分具有连续性，多路召回方法中不同召回方法召回的相似度、热门值不具备可比性，所以我们无法据此来决定每路召回策略的重要性大小。Embedding 召回可以把 Embedding 间的相似度作为唯一的评判标准，因此可以随意限定召回的候选集大小。

最后，在线上服务阶段，Embedding 向量的相似性计算也相对容易实现。通过简单的向量点积或余弦相似度运算就能够得到相似度得分，便于线上的快速召回。

```python
# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item
def embdding_sim(click_df, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的 embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵
        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """
    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    # 建立faiss索引
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_index.search(item_emb_np, topk) # 返回的是列表
    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]): 
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value
    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))   
    return item_sim_dict

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl','rb'))

sim_item_topk = 20
recall_item_num = 10

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)
```

使用 Embedding 计算 item 之间的相似度是为了后续冷启动的时候可以获取未出现在点击数据中的文章，后面有对冷启动专门的介绍，这里简单的说一下 Faiss。


Faiss 使用了 PCA 和 PQ(Product quantization乘积量化)两种技术进行向量压缩和编码，当然还使用了其他的技术进行优化，但是 PCA 和 PQ 是其中核心部分。

