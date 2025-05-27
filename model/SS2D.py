import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm

class SSMConv1D(nn.Module):
    def __init__(self, dim, depth=8):
        super().__init__()
        self.mambalayers = nn.ModuleList([
            nn.Sequential(
                Mamba(d_model=dim),
                RMSNorm(dim)
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.mambalayers:
            x = layer(x)
        return x

class SS2D(nn.Module):
    def __init__(self, channels, depth=8, fusion_method='attention', use_residual=True, diag_mode='both'):
        """
        参数:
            channels: 输入通道数
            depth: SSM 层数
            fusion_method: 融合方法，可选 'average', 'concat', 'attention'
            use_residual: 是否使用残差连接
            diag_mode: 对角线处理模式，可选 'none', 'one', 'both'
        """
        super().__init__()
        self.fusion_method = fusion_method
        self.use_residual = use_residual
        self.diag_mode = diag_mode
        
        # 基础方向处理器
        self.ssm_w = SSMConv1D(channels, depth)  # 水平
        self.ssm_h = SSMConv1D(channels, depth)  # 垂直
        
        # 对角线处理器
        if diag_mode != 'none':
            self.ssm_d1 = SSMConv1D(channels, depth)  # 对角线 \
            if diag_mode == 'both':
                self.ssm_d2 = SSMConv1D(channels, depth)  # 对角线 /
        
        # 计算总方向数
        base_dirs = 4  # 水平2个 + 垂直2个
        diag_dirs = 0
        if diag_mode == 'one':
            diag_dirs = 2  # 对角线 \ 的正向和反向
        elif diag_mode == 'both':
            diag_dirs = 4  # 对角线 \ 和 / 的正向和反向
        
        self.total_dirs = base_dirs + diag_dirs
        
        # 注意力融合模块
        if fusion_method == 'attention':
            self.attention = nn.Sequential(
                nn.Conv2d(channels * self.total_dirs, channels // 4, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(channels // 4, self.total_dirs, kernel_size=1),
                nn.Softmax(dim=1)
            )
        elif fusion_method == 'concat':
            self.projection = nn.Conv2d(channels * self.total_dirs, channels, kernel_size=1)

    def forward(self, x):
        # 设备一致性检查
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        
        B, C, H, W = x.shape
        identity = x if self.use_residual else torch.zeros_like(x)
        
        # 水平方向处理
        seq_w = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        hr = self.ssm_w(seq_w).reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        hl = self.ssm_w(seq_w.flip(1)).flip(1).reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # 垂直方向处理
        seq_h = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        vd = self.ssm_h(seq_h).reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B, C, H, W)
        vu = self.ssm_h(seq_h.flip(1)).flip(1).reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B, C, H, W)
        
        # 对角线方向处理
        directions = [hr, hl, vd, vu]  # 基础方向
        
        if self.diag_mode != 'none':
            # 对角线 \ 处理
            d1_fwd, d1_bwd = self._process_diagonal(x)
            directions.extend([d1_fwd, d1_bwd])
            
            if self.diag_mode == 'both':
                # 对角线 / 处理
                d2_fwd, d2_bwd = self._process_antidiagonal(x)
                directions.extend([d2_fwd, d2_bwd])
        
        # 融合策略
        if self.fusion_method == 'average':
            out = sum(directions) / len(directions)
        elif self.fusion_method == 'concat':
            out = torch.cat(directions, dim=1)
            out = self.projection(out)
        elif self.fusion_method == 'attention':
            fused_features = torch.cat(directions, dim=1)
            attn_weights = self.attention(fused_features)
            # 将注意力权重拆分为各个方向
            weights = torch.chunk(attn_weights, self.total_dirs, dim=1)
            # 加权融合
            out = sum(w * f for w, f in zip(weights, directions))
        
        # 残差连接
        if self.use_residual:
            return out + identity
        else:
            return out

    def _process_diagonal(self, x):
        """处理对角线 \ 方向"""
        B, C, H, W = x.shape
        max_diag_len = H + W - 1
        
        # 收集所有对角线
        diagonals = []
        for i in range(max_diag_len):
            coords = []
            values = []
            for j in range(H):
                k = i - j
                if 0 <= k < W:
                    coords.append((j, k))
                    values.append(x[:, :, j, k].unsqueeze(2))  # (B, C, 1)
            
            if values:
                diag = torch.cat(values, dim=2)  # (B, C, L)
                diagonals.append((diag, coords))
        
        # 处理每个对角线
        processed_diags_fwd = []
        processed_diags_bwd = []
        
        for diag, coords in diagonals:
            L = diag.shape[2]
            if L == 0:
                continue
                
            # 正向处理
            diag_seq_fwd = diag.permute(0, 2, 1)  # (B, L, C)
            processed_fwd = self.ssm_d1(diag_seq_fwd).permute(0, 2, 1)  # (B, C, L)
            
            # 反向处理
            diag_seq_bwd = diag_seq_fwd.flip(1)  # 翻转序列
            processed_bwd = self.ssm_d1(diag_seq_bwd).flip(1).permute(0, 2, 1)  # (B, C, L)
            
            processed_diags_fwd.append((processed_fwd, coords))
            processed_diags_bwd.append((processed_bwd, coords))
        
        # 重构特征图
        out_fwd = torch.zeros_like(x, device=x.device)
        out_bwd = torch.zeros_like(x, device=x.device)
        
        for processed, coords in processed_diags_fwd:
            for idx, (j, k) in enumerate(coords):
                out_fwd[:, :, j, k] = processed[:, :, idx]
        
        for processed, coords in processed_diags_bwd:
            for idx, (j, k) in enumerate(coords):
                out_bwd[:, :, j, k] = processed[:, :, idx]
        
        return out_fwd, out_bwd

    def _process_antidiagonal(self, x):
        """处理对角线 / 方向"""
        B, C, H, W = x.shape
        max_diag_len = H + W - 1
        
        # 收集所有反对角线
        antidiagonals = []
        for i in range(max_diag_len):
            coords = []
            values = []
            for j in range(H):
                k = i - (H - 1 - j)
                if 0 <= k < W:
                    coords.append((j, k))
                    values.append(x[:, :, j, k].unsqueeze(2))  # (B, C, 1)
            
            if values:
                adiag = torch.cat(values, dim=2)  # (B, C, L)
                antidiagonals.append((adiag, coords))
        
        # 处理每个反对角线
        processed_adiags_fwd = []
        processed_adiags_bwd = []
        
        for adiag, coords in antidiagonals:
            L = adiag.shape[2]
            if L == 0:
                continue
                
            # 正向处理
            adiag_seq_fwd = adiag.permute(0, 2, 1)  # (B, L, C)
            processed_fwd = self.ssm_d2(adiag_seq_fwd).permute(0, 2, 1)  # (B, C, L)
            
            # 反向处理
            adiag_seq_bwd = adiag_seq_fwd.flip(1)  # 翻转序列
            processed_bwd = self.ssm_d2(adiag_seq_bwd).flip(1).permute(0, 2, 1)  # (B, C, L)
            
            processed_adiags_fwd.append((processed_fwd, coords))
            processed_adiags_bwd.append((processed_bwd, coords))
        
        # 重构特征图
        out_fwd = torch.zeros_like(x, device=x.device)
        out_bwd = torch.zeros_like(x, device=x.device)
        
        for processed, coords in processed_adiags_fwd:
            for idx, (j, k) in enumerate(coords):
                out_fwd[:, :, j, k] = processed[:, :, idx]
        
        for processed, coords in processed_adiags_bwd:
            for idx, (j, k) in enumerate(coords):
                out_bwd[:, :, j, k] = processed[:, :, idx]
        
        return out_fwd, out_bwd

if __name__ == '__main__':
    batch, C, H, W = 1, 16, 1024, 1024
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型并移至GPU
    models = {
        'base': SS2D(C, depth=8, fusion_method='average', diag_mode='none').to(device),
        'diag_one': SS2D(C, depth=8, fusion_method='attention', diag_mode='one').to(device),
        'diag_both': SS2D(C, depth=8, fusion_method='attention', diag_mode='both').to(device)
    }
    
    # 验证模型设备
    for name, model in models.items():
        print(f"模型 [{name}] 设备: {next(model.parameters()).device}")
    
    # 创建输入并确保在同一设备
    x = torch.randn(batch, C, H, W, device=device)
    
    for name, model in models.items():
        # 确保模型在正确设备上
        model = model.to(device)
        
        # 前向传播
        try:
            with torch.no_grad():  # 仅测试时使用
                y = model(x)
            print(f"[{name}] 输入形状: {x.shape} -> 输出形状: {y.shape}")
            print(f"[{name}] 输出设备: {y.device}")
        except Exception as e:
            print(f"[{name}] 前向传播错误: {e}")
            import traceback
            traceback.print_exc()  # 打印完整堆栈跟踪