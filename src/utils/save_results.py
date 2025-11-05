import torch
import os
import json
import datetime
import matplotlib.pyplot as plt

def draw1(history,run_dir):
    plt.figure(figsize=(14, 6))
            
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
            
    # 绘制 Mean Percentage Rank
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mean_p_rank'], label='Val Mean % Rank', color='orange')
    plt.title('Validation Mean Percentage Rank')
    plt.xlabel('Epoch')
    plt.ylabel('Mean % Rank (Lower is Better)')
    plt.legend()
    plt.grid(True)
            
    plt.suptitle(f"Run: {os.path.basename(run_dir)}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(run_dir, 'curves.png'))
    plt.close() 

def draw2(history,run_dir):
    plt.figure(figsize=(14, 11))
            
    # 绘制训练损失
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_loss_inter'], label='Inter Loss')
    plt.title('Inter Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_loss_intra'], label='Intra Loss')
    plt.title('Intra Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
            
    # 绘制 Mean Percentage Rank
    plt.subplot(2, 2, 2)
    plt.plot(history['val_mean_p_rank'], label='Val Mean % Rank', color='orange')
    plt.title('Validation Mean Percentage Rank')
    plt.xlabel('Epoch')
    plt.ylabel('Mean % Rank (Lower is Better)')
    plt.legend()
    plt.grid(True)
            
    plt.suptitle(f"Run: {os.path.basename(run_dir)}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(run_dir, 'curves.png'))
    plt.close() 

def save_trial_results(run_dir, config, history, best_result, partten = 1):
    """
    将单次试验的结果保存到指定目录。
    """
    # 1. 保存参数
    with open(os.path.join(run_dir, 'params.json'), 'w') as f:
        # 自定义一个转换器以处理非序列化类型
        def default_converter(o):
            if isinstance(o, (torch.device, datetime.datetime)):
                return str(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        
        json.dump(config, f, indent=4, default=default_converter)
    
    # 2. 保存最终指标
    if best_result:
        with open(os.path.join(run_dir, 'best_metrics.json'), 'w') as f:
            json.dump(best_result, f, indent=4)
            
    # 3. 绘制并保存曲线
    try:
        if partten == 1:
            draw1(history,run_dir)
        else:
            draw2(history,run_dir)
        
    except Exception as e:
        print(f"绘制图表时出错: {e}")