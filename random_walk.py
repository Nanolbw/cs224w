import networkx as nx
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


class Node2VecModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(Node2VecModel, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, nodes):
        # print(nodes,self.embeddings(torch.tensor(1)))
        return self.embeddings(nodes)


def random_walk(graph, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if neighbors:
            walk.append(random.choice(neighbors))
        else:
            break
    return walk


def generate_walks(graph, walk_length, num_walks):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(graph, node, walk_length))
    return walks


def generate_positive_pairs(walks):
    positive_pairs = []
    for walk in walks:
        for i in range(len(walk)):
            for j in range(i + 1, len(walk)):
                positive_pairs.append((walk[i], walk[j]))
    return positive_pairs


def generate_negative_samples(graph, num_neg_samples):
    nodes = list(graph.nodes())
    negative_samples = []
    for _ in range(num_neg_samples):
        node1 = random.choice(nodes)
        node2 = random.choice(nodes)
        if node1 != node2 and not graph.has_edge(node1, node2):
            negative_samples.append((node1, node2))
    return negative_samples


# 设置参数
walk_length = 10
num_walks = 20
embedding_dim = 64
num_neg_samples = 5
num_epochs = 5
learning_rate = 0.01

# 创建图
G = nx.karate_club_graph()

# 生成随机游走
walks = generate_walks(G, walk_length, num_walks)

# 生成正样本对
positive_pairs = generate_positive_pairs(walks)

# 生成负样本对
negative_samples = generate_negative_samples(G, len(positive_pairs) * num_neg_samples)

# 创建模型
num_nodes = G.number_of_nodes()
model = Node2VecModel(num_nodes, embedding_dim)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    model.train()

    for pos_pair in positive_pairs:
        node1, node2 = pos_pair
        print('n1',node1)
        pos_emb1 = model(torch.tensor([node1]))
        pos_emb2 = model(torch.tensor([node2]))
        pos_similarity = torch.sum(pos_emb1 * pos_emb2, dim=-1)
        print(torch.sigmoid(pos_similarity),torch.ones_like(pos_similarity))
        pos_loss = criterion(torch.sigmoid(pos_similarity), torch.ones_like(pos_similarity))

        # 负样本损失
        neg_losses = []
        for _ in range(num_neg_samples):
            neg_pair = random.choice(negative_samples)
            neg_node1, neg_node2 = neg_pair
            neg_emb1 = model(torch.tensor([neg_node1]))
            neg_emb2 = model(torch.tensor([neg_node2]))
            neg_similarity = torch.sum(neg_emb1 * neg_emb2, dim=-1)
            neg_loss = criterion(torch.sigmoid(neg_similarity), torch.zeros_like(neg_similarity))
            neg_losses.append(neg_loss)

        neg_loss = torch.mean(torch.stack(neg_losses))

        # 总损失
        loss = pos_loss + neg_loss
        total_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# 获取节点嵌入
node_embeddings = model.embeddings.weight.data.numpy()

# 打印部分节点的嵌入
for node in range(5):
    print(f"Node {node}: {node_embeddings[node][:5]}...")  # 只打印前5维
