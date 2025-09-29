import numpy as np

def calculate_retrieval_metrics(similarity_matrix):
    """
    根据相似度矩阵计算检索评估指标。
    假设第 i 行对应第 i 列是正样本。
    """
    num_queries = similarity_matrix.shape[0]
    ranks = []

    for i in range(num_queries):
        # 获取当前人类视频与所有机器人视频的相似度分数
        scores = similarity_matrix[i, :]
        
        # 获取正样本（正确配对）的分数
        ground_truth_score = scores[i]
        
        # 对所有分数进行降序排序
        sorted_scores = np.sort(scores)[::-1]
        
        # 找到正样本分数在排序后的列表中的位置（排名）
        # np.where 返回一个元组，我们需要第一个数组的第一个元素
        # 排名从1开始，所以需要 +1
        rank = np.where(sorted_scores == ground_truth_score)[0][0] + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # 计算各项指标
    r_at_1 = np.mean(ranks <= 1)
    r_at_5 = np.mean(ranks <= 5)
    r_at_10 = np.mean(ranks <= 10)
    mean_rank = np.mean(ranks)
    mrr = np.mean(1.0 / ranks)
    
    metrics = {
        "R@1": r_at_1,
        "R@5": r_at_5,
        "R@10": r_at_10,
        "Mean Rank": mean_rank,
        "MRR": mrr
    }
    
    return metrics

def calculate_retrieval_metrics_grouped(similarity_matrix, group_size):
    """
    根据相似度矩阵计算检索评估指标，支持分组的正样本。
    
    Args:
        similarity_matrix (np.array): (N, N) 的相似度矩阵。
        group_size (int): 每个正样本组的大小 (n)。
    """
    num_queries = similarity_matrix.shape[0]
    if num_queries % group_size != 0:
        raise ValueError("The total number of samples must be divisible by group_size.")
        
    # 用于存储每个查询的最高排名正样本的排名
    ranks_of_best_positive = []

    # 遍历每个人类视频（查询）
    for i in range(num_queries):
        # 1. 确定当前查询所属的组和该组的正样本索引范围
        group_start_idx = (i // group_size) * group_size
        positive_indices = set(range(group_start_idx, group_start_idx + group_size))
        
        # 2. 获取当前查询与所有机器人视频的相似度分数
        scores = similarity_matrix[i, :]
        
        # 3. 获取按分数降序排列的机器人视频的 *索引*
        sorted_candidate_indices = np.argsort(-scores)
        
        # 4. 找到第一个出现在排序列表中的正样本，记录其排名
        best_rank_for_query = -1
        # enumerate 从 0 开始，所以 rank 是 0-based，我们需要 +1
        for rank, candidate_idx in enumerate(sorted_candidate_indices):
            if candidate_idx in positive_indices:
                best_rank_for_query = rank + 1
                break  # 找到后即可停止
        
        # 如果因为某些原因没有找到（理论上不应发生），可以跳过
        if best_rank_for_query != -1:
            ranks_of_best_positive.append(best_rank_for_query)
            
    ranks = np.array(ranks_of_best_positive)
    
    # 5. 基于最高排名正样本的排名列表，计算各项指标
    r_at_1 = np.mean(ranks <= 1)
    r_at_5 = np.mean(ranks <= 5)
    r_at_10 = np.mean(ranks <= 10)
    mean_rank = np.mean(ranks)
    mrr = np.mean(1.0 / ranks)
    
    metrics = {
        "R@1": r_at_1,
        "R@5": r_at_5,
        "R@10": r_at_10,
        "Mean Best Positive Rank": mean_rank, 
        "MRR": mrr
    }
    
    return metrics

