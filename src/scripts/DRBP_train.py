# Generated from notebook: /Users/khj/Desktop/Project_ongoing/DRBP/code/model/DRBP_train.ipynb
# Generated at: 2025-10-22 19:49:47
# NOTE: Markdown cells are converted to comments.
# IPython magics (e.g., %matplotlib) are kept as-is and may require IPython.

# %% [code] cell 1
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, EsmTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset as TorchDataset, DataLoader
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR

# %% [code] cell 2
class CustomDataset(TorchDataset):
    def __init__(self, dataframe, preprocess_function, label_column='class'):
        self.dataframe = dataframe
        self.preprocess_function = preprocess_function
        self.label_column = label_column

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        example = self.dataframe.iloc[idx]
        inputs = self.preprocess_function({'sequence': example['sequence']})
        
        # inputs가 이미 텐서 형태로 반환된다고 가정하고 수정
        input_ids = inputs['input_ids'].squeeze(0)  # 배치 차원을 제거
        attention_mask = inputs['attention_mask'].squeeze(0)  # 배치 차원을 제거
        labels = torch.tensor(example[self.label_column], dtype=torch.float)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# %% [code] cell 3
class BP_classier:
    def __init__(self, kind_of_work, cv):
        self.kind_of_work = kind_of_work
        self.cv = cv
        
        if self.kind_of_work == 'D':
            self.dup_num = 12

        elif self.kind_of_work == 'R':
            self.dup_num = 50
        
        self.model_name = "facebook/esm2_t33_650M_UR50D"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
        
        # Apply LoRA
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            modules_to_save=['classifier']
        )
        self.lora_model = get_peft_model(self.model, self.lora_config)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.lora_model = nn.DataParallel(self.lora_model)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_model.to(self.device)
        
    def preprocess_function(self, examples):
        return self.tokenizer(examples['sequence'], max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
    
    def data_prepare(self):
        # 데이터 로드
        positive_data = pd.read_csv(f'./New_DRBP/{self.kind_of_work}BP_{self.cv}/{self.kind_of_work}BP_train_{self.cv}.csv')
        negative_data = pd.read_csv(f'./New_DRBP/{self.kind_of_work}BP_{self.cv}/N{self.kind_of_work}BP_train_{self.cv}.csv')

        # 검증 데이터 샘플링
        positive_valid = pd.read_csv(f'./New_DRBP/{self.kind_of_work}BP_{self.cv}/{self.kind_of_work}BP_test_{self.cv}.csv')
        negative_valid = pd.read_csv(f'./New_DRBP/{self.kind_of_work}BP_{self.cv}/N{self.kind_of_work}BP_test_{self.cv}.csv')

        # 검증 데이터 제거 및 증폭
        train_positive_data = positive_data.sample(frac=self.dup_num, replace=True).reset_index(drop=True)
        train_negative_data = negative_data.reset_index(drop=True)

        # 훈련 데이터와 검증 데이터 통합
        train_data = pd.concat([train_positive_data, train_negative_data], ignore_index=True)
        valid_data = pd.concat([positive_valid, negative_valid], ignore_index=True)

        # 데이터셋 저장
        valid_data.to_csv(f'./New_new_DRBP/{self.kind_of_work}BP_{self.cv}/{self.kind_of_work}BP_{self.cv}_valid_dataset.csv', index=None)

        # PyTorch Dataset으로 변환
        train_dataset = CustomDataset(train_data, self.preprocess_function)
        valid_dataset = CustomDataset(valid_data, self.preprocess_function)

        # DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_dataset, batch_size=8)

        return train_loader, valid_loader
    
    def run_model(self):
        self.lora_model.train()
        self.num_epochs = 3
        self.validation_interval = 300
        self.batch_counter = 0
        self.early_stopping_patience = 3
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.best_model_path = f'./New_new_DRBP/{self.kind_of_work}BP_{self.cv}/'
        self.train_loader, self.valid_loader = self.data_prepare()
        self.optimizer = torch.optim.AdamW(self.lora_model.parameters(), lr=2e-5)
        num_training_steps = self.num_epochs * len(self.train_loader)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_training_steps)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(self.num_epochs):
            if self.early_stopping_counter >= self.early_stopping_patience:
                break  # Early stopping이 트리거된 경우 완전히 종료

            total_loss = 0
            correct = 0
            total = 0

            # tqdm을 이용해 훈련 진행 상황을 시각적으로 표시
            for batch in tqdm(self.train_loader):
                # 입력 데이터 준비
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 모델 예측
                outputs = self.lora_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # logits: [batch_size, num_labels]

                # 클래스 1의 로짓만 선택 (양성 클래스)
                logits_class_1 = logits[:, 1]

                # [batch_size, 1] -> [batch_size]
                loss = self.criterion(logits_class_1, labels.float())
                total_loss += loss.item()

                # 역전파 및 옵티마이저 스텝
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # 정확도 계산
                preds = torch.sigmoid(logits_class_1) > 0.5  # 로짓을 0~1 사이 확률로 변환 후 반올림하여 예측 레이블 생성
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # 배치 카운터 증가
                self.batch_counter += 1

                # 500개 배치마다 validation 평가
                if self.batch_counter % self.validation_interval == 0:
                    # 검증 모드로 전환
                    self.lora_model.eval()
                    val_correct = 0
                    val_total = 0
                    val_total_loss = 0
                    all_preds = []
                    all_labels = []
                    all_probs = []

                    with torch.no_grad():
                        for val_batch in self.valid_loader:
                            val_input_ids = val_batch["input_ids"].to(self.device)
                            val_attention_mask = val_batch["attention_mask"].to(self.device)
                            val_labels = val_batch["labels"].to(self.device)

                            val_outputs = self.lora_model(val_input_ids, val_attention_mask)
                            val_logits = val_outputs.logits[:, 1]  # logits 크기 조정

                            val_probs = torch.sigmoid(val_logits)
                            val_preds = val_probs > 0.5

                            val_loss = self.criterion(val_logits, val_labels.float())
                            val_total_loss += val_loss.item()

                            val_correct += (val_preds == val_labels).sum().item()
                            val_total += val_labels.size(0)

                            all_preds.extend(val_preds.cpu().numpy())
                            all_labels.extend(val_labels.cpu().numpy())
                            all_probs.extend(val_probs.cpu().numpy())

                    # Validation metrics
                    val_accuracy = val_correct / val_total
                    precision = precision_score(all_labels, all_preds, average='weighted')
                    recall = recall_score(all_labels, all_preds, average='weighted')
                    f1 = f1_score(all_labels, all_preds, average='weighted')
                    auc = roc_auc_score(all_labels, all_probs)
                    avg_val_loss = val_total_loss / len(self.valid_loader)  # validation loss의 평균 계산


                    print(f"\nValidation Results at Batch {self.batch_counter}:")
                    print(f"Accuracy: {val_accuracy:.4f}")
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                    print(f"F1 Score: {f1:.4f}")
                    print(f"AUC: {auc:.4f}\n")
                    print(f"Avg Validation Loss: {avg_val_loss:.4f}\n")  # Validation Loss 출력

                    # Early Stopping 및 모델 저장
                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self.early_stopping_counter = 0

                        # PEFT 모델 전체 저장 (LoRA 구성 포함)
                        if hasattr(self.lora_model, 'module'):
                            self.lora_model.module.save_pretrained(self.best_model_path)  # DataParallel
                        else:
                            self.lora_model.save_pretrained(self.best_model_path)
                        print(f"Model saved at {self.best_model_path}!")
                    else:
                        self.early_stopping_counter += 1
                        print(f"Early stopping patience: {self.early_stopping_counter}/{self.early_stopping_patience}")

                        if self.early_stopping_counter >= self.early_stopping_patience:
                            print("Early stopping triggered")
                            break  # 학습 종료
                            
                    self.lora_model.train()

            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct / total

            print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# %% [code] cell 9999 - CLI wrapper added for batch runs
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run DRBP training for one or more CV indices")
    parser.add_argument("--kind", "-k", choices=["D", "R"], required=True, help="Task kind: D (DNA) or R (RNA)")
    parser.add_argument("--cv", type=int, help="Run a single CV index (e.g., 3)")
    parser.add_argument("--cv-start", type=int, help="Start of CV range (inclusive)")
    parser.add_argument("--cv-end", type=int, help="End of CV range (inclusive)")
    args = parser.parse_args()

    # Resolve CV set
    cvs: list[int] = []
    if args.cv is not None:
        cvs = [args.cv]
    elif args.cv_start is not None and args.cv_end is not None:
        if args.cv_end < args.cv_start:
            print("cv-end must be >= cv-start", file=sys.stderr)
            sys.exit(2)
        cvs = list(range(args.cv_start, args.cv_end + 1))
    else:
        print("Specify either --cv or both --cv-start and --cv-end", file=sys.stderr)
        sys.exit(2)

    print(f"Running training for kind={args.kind}, CVs={cvs}")
    for cv in cvs:
        print(f"\n=== Start CV {cv} ===")
        trainer = BP_classier(args.kind, cv)
        try:
            trainer.run_model()
        except Exception as e:
            print(f"Error in CV {cv}: {e}")
        finally:
            del trainer
            gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"=== End CV {cv} ===\n")
