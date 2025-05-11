import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time


class PositionalEncoding(nn.Module):
    """增强版位置编码，支持更灵活的输入尺寸"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
    def _get_positional_encoding(self, seq_len):
        pe = torch.zeros(seq_len, self.d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        
        # 处理奇偶维度
        if self.d_model % 2 == 0:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0::2] = torch.sin(position * div_term[:self.d_model//2 + 1])
            pe[:, 1::2] = torch.cos(position * div_term[:self.d_model//2])
        
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        pe = self._get_positional_encoding(x.size(1))
        pe = pe.to(x.device)
        return x + pe[:, :x.size(1), :x.size(2)]


class EnhancedTransformerAutoencoder(nn.Module):
    """增强版Transformer自编码器，专注于提高SPE性能"""
    def __init__(self, input_dim, hidden_dim, nhead=4, num_layers=3, dropout=0.1):
        super(EnhancedTransformerAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入嵌入层 - 使用两层MLP替代单一线性层
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # 确保注意力头数能被隐藏维度整除
        self._adjust_nhead(hidden_dim, nhead)
        
        # Transformer编码器 - 增加层数以提高表达能力
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=self.nhead, 
            dim_feedforward=hidden_dim*4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 增强解码器 - 使用多层MLP替代单一线性层，提高重建能力
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim*3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*3, input_dim)
        )
        
        # 注意力池化层 - 使用自注意力机制替代简单平均池化
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def _adjust_nhead(self, hidden_dim, nhead):
        """确保注意力头数能被隐藏维度整除"""
        if hidden_dim % nhead != 0:
            for i in range(nhead, 0, -1):
                if hidden_dim % i == 0:
                    self.nhead = i
                    print(f"已调整注意力头数为{i}，以匹配隐藏维度{hidden_dim}")
                    return
        self.nhead = nhead
    
    def forward(self, x):
        # 输入形状: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # 重塑为[batch_size, seq_len=1, input_dim]
        x_seq = x.unsqueeze(1)
        
        # 通过嵌入层
        x_emb = self.input_embedding(x_seq)
        
        # 添加位置编码
        x_pos = self.pos_encoder(x_emb)
        
        # 通过Transformer编码器
        encoded = self.transformer_encoder(x_pos)
        
        # 存储编码器特征供后续使用
        encoder_features = encoded
        
        # 解码重建原始输入
        decoded = self.decoder(encoded.squeeze(1))
        
        return decoded, encoder_features
    
    def extract_features(self, x):
        """提取特征用于外部使用（如分类）"""
        batch_size = x.size(0)
        
        # 编码过程
        x_seq = x.unsqueeze(1)
        x_emb = self.input_embedding(x_seq)
        x_pos = self.pos_encoder(x_emb)
        encoded = self.transformer_encoder(x_pos)
        
        # 使用注意力池化提取特征
        attention_weights = self.attention_pooling(encoded)
        features = torch.sum(encoded * attention_weights, dim=1)
        
        return features


def calculate_weighted_spe(model, data, device, importance_weights=None):
    """计算加权SPE指标，考虑变量重要性"""
    model.eval()
    batch_size = 32
    spe_values = []
    
    # 如果未提供权重，则使用均匀权重
    if importance_weights is None:
        importance_weights = torch.ones(data.shape[1], device=device)
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
            # 重建
            reconstructed, _ = model(batch)
            
            # 计算每个样本的加权SPE
            for j in range(len(batch)):
                # 计算每个变量的平方误差并加权
                squared_errors = (batch[j] - reconstructed[j])**2
                weighted_errors = squared_errors * importance_weights
                spe = torch.sum(weighted_errors).cpu().numpy()
                spe_values.append(spe)
    
    return np.array(spe_values)


def adaptive_control_limits(values, false_alarm_target=0.01, min_percentile=0.95, max_percentile=0.999):
    """自适应计算控制限，目标是达到指定的错误报警率"""
    # 使用二分搜索寻找合适的控制限
    low_percentile = min_percentile
    high_percentile = max_percentile
    
    for _ in range(10):  # 最多迭代10次
        mid_percentile = (low_percentile + high_percentile) / 2
        threshold = np.percentile(values, 100 * mid_percentile)
        
        # 计算此阈值下的错误报警率
        false_alarm_rate = np.mean(values > threshold)
        
        # 调整搜索范围
        if false_alarm_rate > false_alarm_target:
            low_percentile = mid_percentile
        else:
            high_percentile = mid_percentile
            
        # 如果已足够接近目标，则停止搜索
        if abs(false_alarm_rate - false_alarm_target) < 0.001:
            break
    
    return np.percentile(values, 100 * mid_percentile)


def train_enhanced_model(X_train, epochs=100, batch_size=32, lr=0.001, hidden_dim=None, validation_split=0.2, model_filename='enhanced_transformer_autoencoder.pth'):
    """训练增强版Transformer自编码器"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置隐藏维度，如果未指定则使用默认值
    input_dim = X_train.shape[1]
    if hidden_dim is None:
        hidden_dim = min(27, input_dim - 1)
    
    # 创建模型
    model = EnhancedTransformerAutoencoder(input_dim, hidden_dim)
    model.to(device)
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 分割训练集和验证集
    train_size = int((1 - validation_split) * len(X_train))
    indices = np.random.permutation(len(X_train))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train_split = X_train[train_indices]
    X_val = X_train[val_indices]
    
    # 创建数据加载器
    train_tensor = torch.tensor(X_train_split, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    val_dataset = torch.utils.data.TensorDataset(val_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for data in train_loader:
            batch = data[0].to(device)
            
            # 前向传播
            reconstructed, _ = model(batch)
            
            # 计算损失（重建误差）
            loss = torch.mean((batch - reconstructed)**2)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                batch = data[0].to(device)
                reconstructed, _ = model(batch)
                loss = torch.mean((batch - reconstructed)**2)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_filename)
        
        # 每10个epoch打印一次进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # 计算训练时间
    train_time = time.time() - start_time
    print(f"训练完成，耗时：{train_time:.2f}秒，最佳验证损失：{best_val_loss:.6f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Enhanced Transformer Autoencoder Training Process')
    plt.legend()
    plt.grid(True)
    plt.savefig('enhanced_transformer_loss.png')
    plt.close()
    
    # 加载最佳模型
    try:
        model.load_state_dict(torch.load(model_filename))
    except Exception as e:
        print(f"警告：无法加载保存的模型 {model_filename}，可能尺寸不匹配: {e}")
        print("继续使用当前模型参数")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    # 创建示例数据（如果需要测试）
    X_train = np.random.randn(500, 52).astype(np.float32)
    
    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # 训练模型
    model, train_losses, val_losses = train_enhanced_model(
        X_train, 
        epochs=50,
        batch_size=32,
        lr=0.001,
        hidden_dim=27,
        validation_split=0.2
    )
    
    print("模型训练和保存完成，可以用于故障检测。") 