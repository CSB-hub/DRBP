# Generated from notebook: /Users/khj/Desktop/Project_ongoing/DRBP/code/model/DRBP_predictin.ipynb
# Generated at: 2025-10-22 19:44:34
# NOTE: Markdown cells are converted to comments.
#       IPython magics (e.g., %matplotlib) are kept as-is and may require IPython.

# %% [code] cell 1
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, EsmTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset as TorchDataset, DataLoader
import pandas as pd
from peft import PeftModel
from tqdm import tqdm
import gc

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
class BP_classier_eval:
    def __init__(self, kind_of_work, cv):
        self.kind_of_work = kind_of_work
        self.cv = cv
        
        self.model_name = "facebook/esm2_t33_650M_UR50D"
        # Use project-relative model path (aligned with training script)
        self.model_path = f"./New_new_DRBP/{self.kind_of_work}BP_{self.cv}"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
        
        self.loaded_model = PeftModel.from_pretrained(model = self.model, model_id = self.model_path)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.loaded_model = nn.DataParallel(self.loaded_model)
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_model.to(self.device)
        
    def preprocess_function(self, examples):
        return self.tokenizer(examples['sequence'], max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
    
    def data_prepare(self):
        # Validation 데이터 로드
        valid_data = pd.read_csv(f'./New_new_DRBP/{self.kind_of_work}BP_{self.cv}/{self.kind_of_work}BP_{self.cv}_valid_dataset.csv')

        # PyTorch Dataset으로 변환
        valid_dataset = CustomDataset(valid_data, self.preprocess_function)

        # DataLoader 생성
        valid_loader = DataLoader(valid_dataset, batch_size=16)

        return valid_loader, valid_data  # valid_data 반환

    def eval_model(self):
        self.loaded_model.eval()
        self.valid_loader, self.valid_data = self.data_prepare()  # valid_data 반환 받음
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
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

                val_outputs = self.loaded_model(val_input_ids, val_attention_mask)
                val_logits = val_outputs.logits[:, 1]  # logits 크기 조정

                val_probs = torch.sigmoid(val_logits)
                val_preds = torch.round(val_probs)

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
        avg_val_loss = val_total_loss / len(self.valid_loader)

        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}\n")
        print(f"Avg Validation Loss: {avg_val_loss:.4f}\n")

        # valid_data에 예측 결과 추가
        self.valid_data['predictions'] = all_preds  # 예측값 열 추가
        self.valid_data['probabilities'] = all_probs  # 확률 열 추가

        # 결과를 CSV로 저장 (ensure dir exists)
        out_dir = Path('New_new_DRBP') / 'RBP_result'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{self.kind_of_work}BP_{self.cv}.csv'
        self.valid_data.to_csv(out_path, index=False)
        print(f"Prediction results saved to {out_path}")

# %% [code] cell 4
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Evaluate DRBP model(s) and write predictions")
    parser.add_argument("--kind", "-k", choices=["D", "R"], required=True, help="Task kind: D (DNA) or R (RNA)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cv", type=int, help="Single CV index to evaluate")
    group.add_argument("--cv-start", type=int, help="Start of CV range (inclusive)")
    parser.add_argument("--cv-end", type=int, help="End of CV range (inclusive, required when using --cv-start)")
    args = parser.parse_args()

    if args.cv is not None:
        cv_list = [args.cv]
    else:
        if args.cv_end is None or args.cv_start is None:
            parser.error("--cv-start and --cv-end must be provided together")
        if args.cv_end < args.cv_start:
            parser.error("--cv-end must be >= --cv-start")
        cv_list = list(range(args.cv_start, args.cv_end + 1))

    print(f"Evaluating kind={args.kind}, CVs={cv_list}")
    for cv in cv_list:
        print(f"\n=== Start CV {cv} ===")
        evaluator = BP_classier_eval(args.kind, cv)
        try:
            evaluator.eval_model()
        except Exception as e:
            print(f"Error in CV {cv}: {e}")
        finally:
            del evaluator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        print(f"=== End CV {cv} ===\n")
