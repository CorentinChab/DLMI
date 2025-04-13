"""In this script, we read a CSV containing IDs, predictions and probabilities,
and create a balanced predictions based on an adapted threshold."""
import pandas as pd

input_filename = 'tta_mean_8.csv'
df = pd.read_csv(input_filename)

id_pred_only = df[['ID', 'Pred']]
id_pred_only.to_csv('id_pred_only.csv', index=False)

pred_counts = df['Pred'].value_counts()
total = len(df)
print("\nCurrent distribution:")
print(pred_counts)
print(f"Fraction of 0s: {pred_counts.get(0, 0) / total:.4f}")
print(f"Fraction of 1s: {pred_counts.get(1, 0) / total:.4f}")

# Find threshold to balance 0s and 1s
sorted_probs = sorted(df['Probability'].tolist())
mid_point = len(sorted_probs) // 2
new_threshold = sorted_probs[mid_point]
print(f"\nNew threshold for balanced classes: {new_threshold:.6f}")

# Apply the new threshold to create regularized predictions
df['RegularizedPred'] = (df['Probability'] >= new_threshold).astype(int)

regularized_df = df[['ID', 'RegularizedPred']]
regularized_df.to_csv('regularized_predictions.csv', index=False)

reg_counts = df['RegularizedPred'].value_counts()
print("\nRegularized distribution:")
print(reg_counts)
print(f"Fraction of 0s: {reg_counts.get(0, 0)/total:.4f}")
print(f"Fraction of 1s: {reg_counts.get(1, 0)/total:.4f}")