import os
import sys
import warnings
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from datetime import datetime

from models.SCLFER import SCLFER
from utils import save_best_model, control_random_seed

def warn(*args, **kwargs):
    pass
warnings.warn = warn

eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/userHome/userhome1/kimhaesung/FER_Models/datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--gpu', type=str, default='0', help='Assign a single GPU by its number.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    
    # DAN
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    
    # 하이퍼 파라미터 정리
    # batch size
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    
    # optimizer
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for AdamW.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    # train test split ratio
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.25, help='Fraction of data to use for validation')
    
    # learning rate scheduler
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'exp'], help='Learning rate scheduler type')
    parser.add_argument('--t_max', type=int, default=100, help='T_max for CosineAnnealingLR scheduler')
    
    # iterations
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations for repeated random sampling')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early stopping patience.')
    
    # Checkpoint directory
    parser.add_argument('--checkpoint_dir', type=str, default='/userHome/userhome1/kimhaesung/FER_Models/FER_Models/DAN/checkpoint/rafdb/mobilefacenet', help='Directory to save checkpoints')
    parser.add_argument('--dataset_name', type=str, default='rafdb', help='Dataset name for checkpoint filename')
    
    # Confusion matrix 관련 파라미터
    parser.add_argument('--emotion_labels', type=str, default='surprise,fear,disgust,happiness,sadness,anger,neutral', help='Emotion class labels for confusion matrix (comma-separated)')
    
    return parser.parse_args()


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, indices, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None, names=['name', 'label'])
        self.file_names = df['name'].values[indices]
        self.labels = df['label'].values[indices] - 1

        self.file_paths = [os.path.join(self.raf_path, 'Image/aligned', f.split(".")[0] + "_aligned.jpg") for f in self.file_names]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]      
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=8, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))
        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

class PartitionLoss(nn.Module):
    def __init__(self):
        super(PartitionLoss, self).__init__()

    def forward(self, x):
        num_head = x.size(1)
        if (num_head > 1):
            var = x.var(dim=1).mean()
            loss = torch.log(1 + num_head / (var + eps))
        else:
            loss = 0
        return loss

def train(train_loader, val_loader, model, criterion_cls, criterion_af, criterion_pt, optimizer, scheduler, 
         device, epochs, patience, iteration, dataset_name, checkpoint_dir, class_labels):
    best_loss = float('inf')
    best_acc = 0
    best_checkpoint_path = None  # 이 변수를 추적하도록 수정
    patience_counter = 0
    best_val_true = None
    best_val_pred = None
    best_epoch = 0  # 최고 성능의 에포크 추적
    best_balanced_acc = 0

    for epoch in tqdm(range(1, epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
            out, feat, heads = model(imgs)

            loss = criterion_cls(out, targets) + criterion_af(feat, targets) + criterion_pt(heads)
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(len(train_loader.dataset))
        running_loss = running_loss / iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss, optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            
            y_true = []
            y_pred = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                out, feat, heads = model(imgs)
                loss = criterion_cls(out, targets) + criterion_af(feat, targets) + criterion_pt(heads)

                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                
                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())
        
            val_loss = val_loss / iter_cnt   
            scheduler.step()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            
            y_true_np = np.concatenate(y_true)
            y_pred_np = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true_np, y_pred_np), 4)

            current_time = datetime.now().strftime('%y%m%d_%H%M%S')
            tqdm.write("[%s] [Epoch %d] Validation accuracy: %.4f. bacc: %.4f. Loss: %.3f" % (current_time, epoch, acc, balanced_acc, val_loss))
            
            if acc > best_acc:
                best_acc = acc
                best_val_true = y_true_np
                best_val_pred = y_pred_np
                best_epoch = epoch  # 최고 성능의 에포크 저장
                best_balanced_acc = balanced_acc
                
                # 정확도가 0.60보다 클 때(충분히 좋은 성능일 때) 체크포인트 저장
                if acc > 0.60:
                    best_checkpoint_path = save_best_model(
                        epoch=epoch,
                        acc=acc,
                        balanced_acc=balanced_acc,
                        model=model,
                        dataset_name=dataset_name,
                        optimizer=optimizer,
                        iteration=iteration,
                        checkpoint_dir=checkpoint_dir,
                        y_true=y_true_np,
                        y_pred=y_pred_np,
                        class_labels=class_labels
                    )
                    tqdm.write(f"Saved new best model at epoch {epoch} with accuracy {acc:.4f}")
            
            tqdm.write("best_acc:" + str(best_acc))
            
            if best_loss >= val_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # 마지막으로 저장된 best_checkpoint_path가 없으면 마지막 에포크 체크포인트를 생성
    if best_checkpoint_path is None and best_acc > 0:
        tqdm.write(f"No checkpoint was saved during training. Saving best model from epoch {best_epoch}")
        best_checkpoint_path = save_best_model(
            epoch=best_epoch,
            acc=best_acc,
            balanced_acc=best_balanced_acc,
            model=model,
            dataset_name=dataset_name,
            optimizer=optimizer,
            iteration=iteration,
            checkpoint_dir=checkpoint_dir,
            y_true=best_val_true,
            y_pred=best_val_pred,
            class_labels=class_labels
        )

    return best_checkpoint_path, best_acc, best_val_true, best_val_pred, best_balanced_acc

def test(test_loader, model, checkpoint_path, criterion_cls, criterion_af, criterion_pt, device,
        class_labels, dataset_name, checkpoint_dir, iteration, epoch, val_acc, val_balanced_acc):
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0

            y_true = []
            y_pred = []

            model.eval()
            for (imgs, targets) in test_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                out, feat, heads = model(imgs)
                loss = criterion_cls(out, targets) + criterion_af(feat, targets) + criterion_pt(heads)

                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())

            running_loss = running_loss / iter_cnt

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

            current_time = datetime.now().strftime('%y%m%d_%H%M%S')
            tqdm.write("[%s] Test accuracy: %.4f. bacc: %.4f. Loss: %.3f" % (current_time, acc, balanced_acc, running_loss))
            
            # 체크포인트 디렉토리가 존재하는지 확인하고 없으면 생성
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            # 테스트 데이터에 대한 confusion matrix 생성 및 저장
            try:
                cm = confusion_matrix(y_true, y_pred)
                test_cm_filename = os.path.join(
                    checkpoint_dir,
                    f"mfnonlychannel_{dataset_name}_iter{iteration+1}_epoch{epoch}_test_cm.png"
                )
                
                plt.figure(figsize=(10, 8))
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.round(cm_norm * 100, 1)
                
                sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', 
                          xticklabels=class_labels, yticklabels=class_labels)
                
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.title(f'Test Confusion Matrix (Iter {iteration+1}, Val Acc: {val_acc:.4f}, Test Acc: {acc:.4f})')
                
                plt.tight_layout()
                plt.savefig(test_cm_filename)
                plt.close()
                
                tqdm.write(f'Test confusion matrix saved at {test_cm_filename}')
            except Exception as e:
                tqdm.write(f"Error creating test confusion matrix: {e}")

            return acc, balanced_acc, running_loss.item(), y_true, y_pred
    except Exception as e:
        tqdm.write(f"Error in test function: {e}")
        return 0, 0, 0, None, None

def run_train_test():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_accuracies = []
    best_accuracies = []
    results = []

    # 체크포인트 디렉토리가 없으면 생성
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    df = pd.read_csv(os.path.join(args.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None, names=['name', 'label'])
    file_names = df['name'].values
    labels = df['label'].values - 1

    ss = ShuffleSplit(n_splits=args.iterations, test_size=args.test_size, random_state=42)
    splits = list(ss.split(file_names, labels))

    for iteration, (train_val_indices, test_indices) in enumerate(splits):
        print(f"Iteration {iteration + 1}/{args.iterations}")
        control_random_seed(iteration)

        model = SCLFER(num_head=args.num_head)
        model.to(device)
        # 모델 파라미터 수 출력
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Total Parameters: %.3fM' % parameters)

        # MobileFaceNet은 112x112 입력을 기대하므로 224x224 이미지를 변환하도록 forward에서 처리
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # 기존 크기 유지
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25)),
        ])

        # train_val_indices 중 75%는 train_indices로, 25%는 val_indices로 나누기 위해 train_test_split 사용
        train_indices, val_indices = train_test_split(train_val_indices, test_size=args.val_size, random_state=iteration)

        train_dataset = RafDataSet(args.raf_path, phase='train', indices=train_indices, transform=data_transforms)
        print('Train set size:', train_dataset.__len__())

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=args.workers,
                                                  shuffle=True,
                                                  pin_memory=True)

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # 기존 크기 유지
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = RafDataSet(args.raf_path, phase='validation', indices=val_indices, transform=val_transforms)
        print('Validation set size:', val_dataset.__len__())

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.workers,
                                                shuffle=False,
                                                pin_memory=True)

        test_dataset = RafDataSet(args.raf_path, phase='test', indices=test_indices, transform=val_transforms)
        print('Test set size:', test_dataset.__len__())

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 shuffle=False,
                                                 pin_memory=True)

        criterion_cls = torch.nn.CrossEntropyLoss()
        criterion_af = AffinityLoss(device)
        criterion_pt = PartitionLoss()

        params = list(model.parameters()) + list(criterion_af.parameters())
        
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        if args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
        elif args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif args.lr_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            
        # 감정 클래스 레이블 파싱
        class_labels = args.emotion_labels.split(',')
        
        try:
            best_checkpoint_path, best_acc, best_val_true, best_val_pred, best_val_balanced_acc = train(
                train_loader=train_loader, 
                val_loader=val_loader, 
                model=model, 
                criterion_cls=criterion_cls, 
                criterion_af=criterion_af, 
                criterion_pt=criterion_pt, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                device=device, 
                epochs=args.epochs, 
                patience=args.early_stopping_patience, 
                iteration=iteration,
                dataset_name=args.dataset_name,
                checkpoint_dir=args.checkpoint_dir,
                class_labels=class_labels
            )

            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                try:
                    checkpoint = torch.load(best_checkpoint_path)
                    epoch_num = checkpoint.get('epoch', 0)  # 안전하게 가져오기
                    
                    test_acc, test_balanced_acc, test_running_loss, test_true, test_pred = test(
                        test_loader=test_loader,
                        model=model, 
                        checkpoint_path=best_checkpoint_path, 
                        criterion_cls=criterion_cls, 
                        criterion_af=criterion_af, 
                        criterion_pt=criterion_pt, 
                        device=device,
                        class_labels=class_labels,
                        dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir,
                        iteration=iteration,
                        epoch=epoch_num,
                        val_acc=best_acc,
                        val_balanced_acc=best_val_balanced_acc
                    )
                    
                    all_accuracies.append(test_acc)
                    best_accuracies.append(best_acc)
                    results.append([iteration + 1, test_acc, test_balanced_acc, test_running_loss])
                    print(f"Test Accuracy: {test_acc}, Test Balanced Accuracy: {test_balanced_acc}, Test Running Loss: {test_running_loss}")
                except Exception as e:
                    print(f"Error during testing: {e}")
                    print(f"Skipping iteration {iteration + 1}")
            else:
                print(f"No valid checkpoint found for iteration {iteration + 1}. Skipping testing.")
        except Exception as e:
            print(f"Error during training iteration {iteration + 1}: {e}")
            print(f"Skipping iteration {iteration + 1}")

    # 결과 저장
    if results:
        results_df = pd.DataFrame(results, columns=['Iteration', 'Test Accuracy', 'Balanced Accuracy', 'Loss'])
        results_df.to_csv('DAN_MobileNetFace_test_results.csv', index=False)  # 결과 파일명 변경

        print("\nBest Accuracies over all iterations:")
        for i, acc in enumerate(best_accuracies, 1):
            print(f"Iteration {i}: {acc:.4f}")

        if all_accuracies:
            mean_accuracy = np.mean(all_accuracies)
            std_accuracy = np.std(all_accuracies)
            print(f"\nMean Test Accuracy over {len(all_accuracies)} iterations: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        else:
            print("\nNo results were collected from any iteration.")
            
if __name__ == "__main__":
    try:
        run_train_test()
    except Exception as e:
        print(f"Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()