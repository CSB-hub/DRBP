# Generated from notebook: /Users/khj/Desktop/Project_ongoing/DRBP/code/model/DRBP_att_score.ipynb
# Generated at: 2025-10-22 19:44:34
# NOTE: Markdown cells are converted to comments.
#       IPython magics (e.g., %matplotlib) are kept as-is and may require IPython.

# %% [code] cell 1
import torch
from transformers import AutoModelForSequenceClassification, EsmTokenizer
import pandas as pd
from peft import PeftModel
import gc
from pathlib import Path

# %% [code] cell 2
def calculate_vatp_scores(model, input_ids, attention_mask):
    device = input_ids.device
    # 모델에서 attention과 hidden_states를 출력하도록 설정
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True)
    
    attentions = outputs.attentions
    hidden_states = outputs.hidden_states
    
    vatp_scores = []
    
    for layer in range(len(attentions)):
        layer_scores = []
        
        value_vectors = model.model.esm.encoder.layer[layer].attention.self.value(hidden_states[layer])
        
        for head in range(attentions[layer].shape[1]):
            attention_scores = attentions[layer][:, head, 0, :]
            value_norms = torch.norm(value_vectors[:, :, head * 64:(head + 1) * 64], p=1, dim=2)
            head_vatp_scores = attention_scores * value_norms
            layer_scores.append(head_vatp_scores)
        
        # 각 레이어의 VATP 점수 계산
        layer_vatp_scores = torch.stack(layer_scores).mean(dim=0)
        vatp_scores.append(layer_vatp_scores)
    
    # 모든 레이어의 VATP 점수를 평균하여 최종 점수 계산
    final_vatp_scores = torch.stack(vatp_scores).mean(dim=0)
    return final_vatp_scores

# %% [code] cell 3
def run_vatp_for_cv(kind_of_work: str, cv: int, pad_len: int = 40, prefix: str = "251010") -> Path:
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    check = f"./New_new_DRBP/{kind_of_work}BP_{cv}"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    loaded_model = PeftModel.from_pretrained(model, check)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_model.to(device)
    loaded_model.eval()

    sequence = pd.read_csv(f"{check}/{kind_of_work}BP_{cv}_valid_dataset.csv")

    # 앞쪽에 pad_len 길이의 PAD 토큰 추가
    pad_token = tokenizer.pad_token
    input_sequence = sequence["sequence"].apply(lambda seq: pad_token * pad_len + seq)

    sequence_ids = sequence["ID"]
    final_data: dict[str, list] = {}

    # 각 시퀀스에 대해 VATP 점수 계산
    for i in range(len(input_sequence)):
        inputs = tokenizer(
            input_sequence[i],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            vatp_scores = calculate_vatp_scores(loaded_model, inputs["input_ids"], inputs["attention_mask"])
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            token_col_name = f"{sequence_ids[i]}"
            vatp_col_name = f"VATP_{sequence_ids[i]}"

            token_list = []
            vatp_score_list = []
            for token, score in zip(tokens, vatp_scores[0]):
                token_list.append(token)
                vatp_score_list.append(score.item())

            final_data[token_col_name] = token_list
            final_data[vatp_col_name] = vatp_score_list

    df_final = pd.DataFrame(final_data)
    out_dir = Path("./New_DRBP") / f"{kind_of_work}BP_VATP_result_{prefix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_VATP_score_{pad_len}PAD_{kind_of_work}NA{cv}.csv"
    df_final.to_csv(out_path, index=None)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute VATP token scores for valid set")
    parser.add_argument("--kind", "-k", choices=["D", "R"], required=True, help="Task kind: D (DNA) or R (RNA)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cv", type=int, help="Single CV index")
    group.add_argument("--cv-start", type=int, help="Start of CV range (inclusive)")
    parser.add_argument("--cv-end", type=int, help="End of CV range (inclusive) when using --cv-start")
    parser.add_argument("--pad-len", type=int, default=40, help="Number of PAD tokens to prefix")
    parser.add_argument("--prefix", type=str, default="251010", help="Prefix for output folder and filename")
    args = parser.parse_args()

    if args.cv is not None:
        cvs = [args.cv]
    else:
        if args.cv_end is None or args.cv_start is None:
            parser.error("--cv-start and --cv-end must be provided together")
        if args.cv_end < args.cv_start:
            parser.error("--cv-end must be >= --cv-start")
        cvs = list(range(args.cv_start, args.cv_end + 1))

    print(f"VATP scoring for kind={args.kind}, CVs={cvs}, pad_len={args.pad_len}, prefix={args.prefix}")
    for cv in cvs:
        try:
            out = run_vatp_for_cv(args.kind, cv, pad_len=args.pad_len, prefix=args.prefix)
            print(f"OK CV {cv} -> {out}")
        except Exception as e:
            print(f"ERR CV {cv}: {e}")
