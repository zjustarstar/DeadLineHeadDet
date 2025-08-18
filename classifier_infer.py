from PIL import Image
import torch
import numpy as np
import os
from torchvision import transforms
from classifier_network import ResNetClassifier


def create_data_transforms():
    """
    创建数据增强转换
    """
    # 获取区域大小
    input_size = 100

    # 验证集转换（不包含数据增强）
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return val_transform


def infer_main(model, trans, image, device):
    """
    推理模型
    Args:

    Returns:
        训练历史记录
    """

    # 将模型移动到设备
    model = model.to(device)
    model.eval()  # 确保模型处于评估模式
        
    # 推理
    with torch.no_grad():
        inputs = trans(image)

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(np.array(inputs))

        if inputs.dim() == 3:  # 如果是 CHW 格式
            inputs = inputs.unsqueeze(0)  # 添加 batch 维度

        # 移动到设备
        inputs = inputs.to(device)

        # 前向传播
        outputs = model(inputs)
        
        # 打印原始输出和概率分布
        print("原始输出:", outputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        print("概率分布:", probabilities)
        
        for i, prob in enumerate(probabilities[0]):
            print(f"类别 {i}: {prob.item():.4f} ({prob.item()*100:.2f}%)")

        # 统计
        _, predicted = torch.max(outputs, 1)

    return predicted, probabilities


def load_model(model_path, device):
    # 创建模型
    model = ResNetClassifier(num_classes=7)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return model
    
    try:
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        
        # # 打印状态字典的键
        # print(f"\n状态字典包含以下键:")
        # for key in state_dict.keys():
        #     print(f"  - {key}")
        
        # 检查状态字典中的权重
        # if 'resnet.fc.weight' in state_dict:
        #     fc_weight = state_dict['resnet.fc.weight']
        #     print(f"\n最后一层权重统计(从状态字典):\n")
        #     print(f"形状: {fc_weight.shape}")
        #     print(f"均值: {fc_weight.mean().item():.6f}")
        #     print(f"标准差: {fc_weight.std().item():.6f}")
        #     print(f"最大值: {fc_weight.max().item():.6f}")
        #     print(f"最小值: {fc_weight.min().item():.6f}")
        #
        #     # 检查是否所有权重都相同
        #     if torch.allclose(fc_weight[0], fc_weight[1], atol=1e-5) and torch.allclose(fc_weight[0], fc_weight[2], atol=1e-5):
        #         print("警告: 最后一层的权重几乎完全相同，这可能导致所有类别的预测结果一致!")
        
        # 加载状态字典到模型
        model.load_state_dict(state_dict)
        print(f"\n成功加载模型: {model_path}")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
    
    return model


def main():
    # 设置多个测试样本，从不同类别中选择
    test_images = [
        "sample\\n\\0721_171625_400.png",       # n类别样本
        "sample\\ss\\0721_172011_303.png",     # ss类别样本
        "sample\\other\\0721_171025_664.png"   # other类别样本
    ]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model_path = ".\\models\\classifier.pth"
    model = load_model(model_path, device)

    # 检查模型参数
    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params}")
    print(f"可训练参数数量: {trainable_params}")
    
    # 检查最后一层权重
    fc_weights = model.resnet.fc.weight.data
    print("\n最后一层权重统计:")
    print(f"权重形状: {fc_weights.shape}")
    print(f"权重均值: {fc_weights.mean().item():.6f}")
    print(f"权重标准差: {fc_weights.std().item():.6f}")
    print(f"权重最大值: {fc_weights.max().item():.6f}")
    print(f"权重最小值: {fc_weights.min().item():.6f}")

    trans = create_data_transforms()

    # 对每个测试样本进行推理
    for i, image_path in enumerate(test_images):
        print(f"\n\n测试样本 {i+1}: {image_path}")
        try:
            img = Image.open(image_path).convert('RGB')
            class_id, probabilities = infer_main(model, trans, img, device=device)
            
            # 获取类别名称
            if class_id.item() == 0:
                class_name = "N"
            elif class_id.item() == 1:
                class_name = "ss"
            elif class_id.item() == 2:
                class_name = "wv"
            elif class_id.item() == 3:
                class_name = "t"
            else:
                class_name = "other"
                
            print(f"预测类别ID: {class_id.item()}")
            print(f"预测类别名称: {class_name}")
            
            # 打印所有类别的概率
            print("\n所有类别的概率:")
            class_names = ["N", "ss", "wv", "t", "other"]
            for j, prob in enumerate(probabilities[0]):
                print(f"{class_names[j]}: {prob.item():.6f} ({prob.item()*100:.2f}%)")
                
        except Exception as e:
            print(f"处理样本时出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()