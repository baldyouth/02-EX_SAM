import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2, smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # 计算预测概率
        y_pred = torch.sigmoid(y_pred)  # 确保预测值在 [0, 1] 范围内
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)  # 计算 p_t
        
        # 计算 Focal Loss
        loss = - self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + self.smooth)
        return torch.mean(loss)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # 将预测值转换为 [0, 1] 范围内的概率
        y_pred = torch.sigmoid(y_pred)

        # 计算真阳性、假阳性和假阴性
        tp = torch.sum(y_true * y_pred)
        fp = torch.sum((1 - y_true) * y_pred)
        fn = torch.sum(y_true * (1 - y_pred))
        
        # 计算 Tversky Loss
        loss = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - loss  # 最大化 Tversky Score，最小化损失

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, smooth=smooth)
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, y_pred, y_true):
        focal_loss = self.focal_loss(y_pred, y_true)
        tversky_loss = self.tversky_loss(y_pred, y_true)
        
        # 返回两个损失的和
        return focal_loss + tversky_loss

# -------------------------------------------------------------------------------------------------
# 测试类
class LossTest:
    def __init__(self, loss_fn, input_shape=(1, 1, 256, 256)):
        self.loss_fn = loss_fn
        self.input_shape = input_shape

    def generate_fake_data(self):
        """
        生成真实标签和预测值完全相同的值
        """
        # 生成完全相同的预测值和真实标签
        y_pred = torch.zeros(self.input_shape)  # 假设全是1
        y_true = torch.ones(self.input_shape)  # 假设全是1
        return y_pred, y_true

    def run_test(self):
        """
        运行测试
        """
        y_pred, y_true = self.generate_fake_data()

        # 计算损失
        loss_value = self.loss_fn(y_pred, y_true)

        # 输出损失值
        print(f"Calculated Loss: {loss_value.item()}")

        # 验证两个输入值是否相同
        if torch.equal(y_pred, y_true):
            print("Test Passed: The inputs are identical.")
        else:
            print("Test Failed: The inputs are not identical.")



if __name__ == '__main__':
    import torch
    import segmentation_models_pytorch as smp

    # 初始化 FocalLoss，mode='binary' 表示二分类
    focal_loss = smp.losses.FocalLoss(mode='binary')
    tversky_loss = smp.losses.TverskyLoss(mode='binary')

    N, H, W = 2, 128, 128
    y_true = torch.zeros((N, 1, H, W)).float()
    y_pred = torch.ones((N, 1, H, W)).float()

    print("y_pred 和 y_true 是否相同:", torch.allclose(y_pred, y_true))

    focalLoss = focal_loss(y_pred, y_true)
    print(f"Focal Loss: {focalLoss.item()}")

    tverskyLoss = tversky_loss(y_pred, y_true)
    print(f"Tversky Loss: {tverskyLoss.item()}")


