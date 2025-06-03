import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualization
plt.style.use('ggplot')  # Using a valid matplotlib style
sns.set_theme()  # Using seaborn's default theme

# Read the CSV file
df = pd.read_csv('MoCA-Video User Study (Responses).csv')

# Define the metrics and approaches
metrics = ['Blending Quality (BQ)', 'Video Consistency (VC)', 'Character Consistency (CC)', 'Overall Quality (OQ)']
approaches = ['MoCA-Video', 'AnimateDiffV2V', 'FreeBlend + DynamiCrafter']  # Updated MoCA to MoCA-Video

# Function to calculate weighted average
def calculate_weighted_average(ratings, weights):
    return np.average(ratings, weights=weights)

# Initialize results dictionary
results = {}

# Process each approach and metric combination
for approach in approaches:
    for metric in metrics:
        # Get all columns that match this approach and metric
        # Handle the name changes in the column matching
        search_name = 'AnimateDiff' if approach == 'AnimateDiffV2V' else 'FreeBlend' if approach == 'FreeBlend + DynamiCrafter' else 'MoCA' if approach == 'MoCA-Video' else approach
        cols = [col for col in df.columns if search_name in col and metric in col]
        
        # Get the ratings and weights (participant levels)
        ratings = []
        weights = []
        
        for col in cols:
            # Get the ratings for this column
            col_ratings = df[col].dropna().values
            # Get the corresponding participant levels (from the first column)
            col_weights = df.iloc[:len(col_ratings), 0].values
            
            ratings.extend(col_ratings)
            weights.extend(col_weights)
        
        # Calculate weighted average
        weighted_avg = calculate_weighted_average(ratings, weights)
        results[f"{approach} - {metric}"] = weighted_avg

# Print results in a formatted table
print("\nWeighted Average Results (5 being highest weight):")
print("-" * 80)
print(f"{'Approach':<30} {'Metric':<30} {'Weighted Average':<15}")
print("-" * 80)

for key, value in results.items():
    approach, metric = key.split(" - ")
    print(f"{approach:<30} {metric:<30} {value:.2f}")

# Calculate average for each approach
print("\nAverage Scores by Approach:")
print("-" * 40)
for approach in approaches:
    approach_scores = [v for k, v in results.items() if approach in k]
    avg_score = np.mean(approach_scores)
    print(f"{approach:<30} {avg_score:.2f}")

# Calculate average for each metric
print("\nAverage Scores by Metric:")
print("-" * 40)
for metric in metrics:
    metric_scores = [v for k, v in results.items() if metric in k]
    avg_score = np.mean(metric_scores)
    print(f"{metric:<30} {avg_score:.2f}")

# Create visualizations
plt.figure(figsize=(20, 15))

# Set font sizes
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

# Create a 2x2 grid of subplots
for idx, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, idx)
    
    # Get scores for this metric
    metric_scores = []
    for approach in approaches:
        score = results[f"{approach} - {metric}"]
        metric_scores.append(score)
    
    # Create bar plot
    bars = plt.bar(approaches, metric_scores)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=16, fontweight='bold')
    
    # Customize plot
    plt.title(f'{metric} Comparison', pad=20, fontsize=20, fontweight='bold')
    plt.ylabel('Weighted Average Score', fontsize=18, fontweight='bold')
    plt.ylim(0, 5)  # Set y-axis limit to match rating scale
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability and make them bold
    plt.xticks(rotation=15, ha='right')
    for label in plt.gca().get_xticklabels():
        label.set_fontweight('bold')
    
    # Make y-axis labels bold
    for label in plt.gca().get_yticklabels():
        label.set_fontweight('bold')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('metric_comparison_plots.png', dpi=300, bbox_inches='tight')
print("\nPlot has been saved as 'metric_comparison_plots.png'") 