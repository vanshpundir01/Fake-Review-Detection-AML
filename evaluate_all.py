import subprocess
import sys
import re

PYTHON = sys.executable

def run_script(script):
    print(f"\n{'='*55}\nRunning {script}\n{'='*55}")
    result = subprocess.run([PYTHON, script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("ERRORS/WARNINGS:")
        print(result.stderr)
    return result.stdout

def extract(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

def fmt(x):
    return f"{x:.4f}" if x is not None else "N/A"

print("Starting complete evaluation...")
train_out = run_script("train.py")
test_out = run_script("test.py")
attack_out = run_script("attack_test.py")
def_train_out = run_script("train_defense.py")
def_test_out = run_script("test_defense.py")

train_acc = extract(r"Training Accuracy:\s*([0-9.]+)", train_out)
test_acc = extract(r"Accuracy:\s*([0-9.]+)", test_out)
attack_acc = extract(r"Accuracy on Adversarial Data:\s*([0-9.]+)", attack_out)
def_train_acc = extract(r"Defense Training Accuracy:\s*([0-9.]+)", def_train_out)
def_clean = extract(r"Accuracy \(Clean\):\s*([0-9.]+)", def_test_out)
def_attack = extract(r"Accuracy \(Attacked\):\s*([0-9.]+)", def_test_out)

print("\n" + "="*55)
print("FINAL ACCURACY REPORT")
print("="*55)
print(f"Standard Train Accuracy:   {fmt(train_acc)}")
print(f"Standard Test Accuracy:    {fmt(test_acc)}")
print(f"Standard Attack Accuracy:  {fmt(attack_acc)}")
print("-"*55)
print(f"Defense Train Accuracy:    {fmt(def_train_acc)}")
print(f"Defense Clean Accuracy:    {fmt(def_clean)}")
print(f"Defense Attack Accuracy:   {fmt(def_attack)}")
print("="*55)
