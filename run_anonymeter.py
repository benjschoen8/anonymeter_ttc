import pandas as pd
import subprocess
import sys
from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
from sklearn.model_selection import train_test_split

# ----------------------
# 1. Dataset Splitting
# ----------------------

def split_dataset(input_csv, ori_csv, control_csv, test_size=0.3, seed=42):
    df = pd.read_csv(input_csv)
    ori, control = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    ori.to_csv(ori_csv, index=False)
    control.to_csv(control_csv, index=False)
    print(f"[✓] Split {input_csv} → {ori_csv} (train), {control_csv} (control)")

# ----------------------
# 2. Run Differential Privacy
# ----------------------

def run_dp(input_csv, output_csv):
    print(f"[→] Running DP on {input_csv} to generate {output_csv}...")
    subprocess.run([sys.executable, "apply_dp.py", input_csv, output_csv], check=True)
    print(f"[✓] DP done: {output_csv}")

# ----------------------
# 3. Evaluator Functions
# ----------------------

def run_singling_out(df_ori, df_control, df_synth):
    print("\n→ Evaluating Singling Out risk...")
    evaluator = SinglingOutEvaluator(
        ori=df_ori,
        control=df_control,
        syn=df_synth,
        n_attacks=32,
        n_cols=1  # required due to internal anonymeter constraints
    )
    evaluator.evaluate()
    risk = evaluator.risk()
    print(f"[✓] Estimated Singling Out Risk: {risk}")

def run_linkability(df_ori, df_control, df_synth):
    print("\n→ Evaluating Linkability risk...")
    evaluator = LinkabilityEvaluator(
        ori=df_ori,
        control=df_control,
        syn=df_synth,
        n_attacks=32,
        aux_cols=["name", "size", "pixels"],
    )
    evaluator.evaluate()
    risk = evaluator.risk()
    print(f"[✓] Estimated Linkability Risk: {risk}")

def run_inference(df_ori, df_control, df_synth):
    print("\n→ Evaluating Inference risk...")
    evaluator = InferenceEvaluator(
        ori=df_ori,
        control=df_control,
        syn=df_synth,
        n_attacks=32,
        aux_cols=["name", "size"],
        secret="pixels"
    )
    evaluator.evaluate()
    risk = evaluator.risk()
    print(f"[✓] Estimated Inference Risk: {risk}")

# ----------------------
# 4. Run Anonymeter
# ----------------------

def run_anonymeter(ori_csv, control_csv, syn_csv):
    df_ori = pd.read_csv(ori_csv)
    df_control = pd.read_csv(control_csv)
    df_synth = pd.read_csv(syn_csv)

    # Optional: drop labels if not needed
    # df_ori = df_ori.drop(columns=["label"])
    # df_control = df_control.drop(columns=["label"])
    # df_synth = df_synth.drop(columns=["label"])

    print("\n[🔍] Running Anonymeter Privacy Risk Evaluations:")
    run_singling_out(df_ori, df_control, df_synth)
    run_linkability(df_ori, df_control, df_synth)
    run_inference(df_ori, df_control, df_synth)

# ----------------------
# 5. Main Program
# ----------------------

def main():
    input_csv = "images.csv"
    ori_csv = "ori.csv"
    control_csv = "control.csv"
    syn_csv = "syn.csv"

    split_dataset(input_csv, ori_csv, control_csv)
    run_dp(ori_csv, syn_csv)
    run_anonymeter(ori_csv, control_csv, syn_csv)

if __name__ == "__main__":
    main()

