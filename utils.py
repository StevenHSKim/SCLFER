import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random

# 랜덤 시드를 고정하는 함수
def control_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(cm, classes, iteration, epoch, acc, balanced_acc, checkpoint_dir, dataset_name):
    """
    주어진 confusion matrix를 시각화하고 파일로 저장하는 함수
    """
    try:
        plt.figure(figsize=(10, 8))
        
        # confusion matrix 시각화 (normalize=True로 설정하여 비율로 표시)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.round(cm_norm * 100, 1)  # 백분율로 변환하고 소수점 한자리까지 표시
        
        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix (Iter {iteration+1}, Epoch {epoch}, Acc: {acc:.4f}, Bacc: {balanced_acc:.4f})')
        
        # 체크포인트 디렉토리가 존재하는지 확인하고 없으면 생성
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # 파일로 저장
        cm_filename = os.path.join(
            checkpoint_dir,
            f"mfnonlychannel_{dataset_name}_iter{iteration+1}_epoch{epoch}_cm.png"  # 파일명을 MobileFaceNet 적용에 맞게 변경
        )
        plt.tight_layout()
        plt.savefig(cm_filename)
        plt.close()
        
        return cm_filename
    except Exception as e:
        tqdm.write(f"Error creating confusion matrix: {e}")
        return None

def save_best_model(epoch, acc, balanced_acc, model, dataset_name, optimizer, optimizer_center, iteration, checkpoint_dir, 
                   y_true=None, y_pred=None, class_labels=None):
    """
    best model의 checkpoint를 저장하는 함수
    동일한 iteration 내에서 이전의 checkpoint와 confusion matrix는 삭제
    따라서 각 iteration 별로 한 개의 best model checkpoint와 confusion matrix만 저장
    """
    # 체크포인트 디렉토리가 존재하는지 확인하고 없으면 생성
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # 이전 checkpoint와 confusion matrix 찾기 및 삭제
    pattern = f"centerloss_mfnonlychannel_{dataset_name}_iter{iteration+1}_epoch"
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(pattern) and (filename.endswith(".pth") or filename.endswith("_cm.png")):
            # 테스트 confusion matrix는 유지 (테스트 cm 파일은 삭제하지 않음)
            if "_test_cm.png" in filename:
                continue
                
            previous_file = os.path.join(checkpoint_dir, filename)
            try:
                if os.path.exists(previous_file):
                    os.remove(previous_file)
                    tqdm.write(f"Removed previous file: {previous_file}")
            except Exception as e:
                tqdm.write(f"Warning: Could not remove file {previous_file}: {e}")
                
    # 새로운 checkpoint 저장
    best_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"centerloss_mfnonlychannel_{dataset_name}_iter{iteration+1}_epoch{epoch}_acc{acc}_bacc{balanced_acc}.pth"
    )
    
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_center_state_dict': optimizer_center.state_dict(),  # Center Loss 옵티마이저 상태 저장 추가
            'accuracy': acc,
            'balanced_accuracy': balanced_acc
        }, best_checkpoint_path)
        tqdm.write(f"Saved checkpoint to {best_checkpoint_path}")
    except Exception as e:
        tqdm.write(f"Error saving checkpoint: {e}")
        return None
    
    # y_true와 y_pred가 제공된 경우 confusion matrix 생성 및 저장
    if y_true is not None and y_pred is not None and class_labels is not None:
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.round(cm_norm * 100, 1)
            
            sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', 
                      xticklabels=class_labels, yticklabels=class_labels)
            
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title(f'Validation Confusion Matrix (Iter {iteration+1}, Acc: {acc:.4f})')
            
            cm_filename = os.path.join(
                checkpoint_dir,
                f"centerloss_mfnonlychannel_{dataset_name}_iter{iteration+1}_epoch{epoch}_cm.png"
            )
            
            plt.tight_layout()
            plt.savefig(cm_filename)
            plt.close()
            
            tqdm.write(f'New best model and confusion matrix saved at {best_checkpoint_path} and {cm_filename}')
        except Exception as e:
            tqdm.write(f"Error creating confusion matrix: {e}")
    else:
        tqdm.write(f'New best model saved at {best_checkpoint_path}')
        
    return best_checkpoint_path