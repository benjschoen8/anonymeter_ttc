import pandas as pd
import numpy as np
import sys
import ast

def laplace_mechanism(value, sensitivity=1.0, epsilon=1.0):
    noise = np.random.laplace(0, sensitivity / epsilon)
    return value + noise

def apply_dp_to_pixels(pixel_str, epsilon):
    try:
        # Safely convert string to Python list
        matrix = ast.literal_eval(pixel_str)
        matrix = np.array(matrix, dtype=np.float32)

        # Add Laplace noise
        noisy = matrix + np.random.laplace(loc=0, scale=1.0/epsilon, size=matrix.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        # Return back as string for CSV
        return str(noisy.tolist())
    except Exception as e:
        print(f"[!] Failed to process image matrix: {e}")
        return pixel_str  # fallback to original if error

def apply_dp_to_csv(input_csv, output_csv, epsilon=1.0):
    df = pd.read_csv(input_csv)
    
    if 'pixels' not in df.columns:
        print("[✗] Input CSV must contain a 'pixels' column.")
        sys.exit(1)

    df['pixels'] = df['pixels'].apply(lambda x: apply_dp_to_pixels(x, epsilon))

    df.to_csv(output_csv, index=False)
    print(f"[✓] DP-applied CSV saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python apply_dp.py <input_csv> <output_csv> [epsilon]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    epsilon = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    apply_dp_to_csv(input_csv, output_csv, epsilon=epsilon)

