import torch
import numpy as np
from model import PSLAModel
from preprocessing import ParkinsonsDataset, custom_collate, AudioPreprocessor
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def plot_roc_curve(labels, probs, save_path='roc_curve.png'):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(labels, probs, save_path='pr_curve.png'):
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(save_path)
    plt.close()

def plot_confidence_distribution(probs, labels, save_path='confidence_dist.png'):
    plt.figure(figsize=(10, 6))
    plt.hist(probs[labels == 0], bins=30, alpha=0.5, label='Healthy', color='green')
    plt.hist(probs[labels == 1], bins=30, alpha=0.5, label='PD', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Distribution of Model Confidence')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def print_system_info():
    print("\n=== System Information ===")
    system_info = [
        ["PyTorch Version", torch.__version__],
        ["CUDA Available", torch.cuda.is_available()],
        ["CUDA Device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"],
        ["Device", "cuda" if torch.cuda.is_available() else "cpu"]
    ]
    print(tabulate(system_info, headers=["Parameter", "Value"], tablefmt="grid"))

def print_preprocessing_params(preprocessor):
    print("\n=== Audio Preprocessing Parameters ===")
    preprocessing_params = [
        ["Sample Rate", preprocessor.sample_rate],
        ["Number of Mel Bands", preprocessor.n_mels],
        ["FFT Size", preprocessor.mel_spec.n_fft],
        ["Hop Length", preprocessor.hop_length],
        ["Window Length", preprocessor.mel_spec.win_length],
        ["Frequency Range", f"0-{preprocessor.mel_spec.f_max}Hz"],
        ["Power", preprocessor.mel_spec.power],
        ["Recording Configs", ""],
    ]
    
    # Add recording configurations
    for recording_type, config in preprocessor.recording_configs.items():
        preprocessing_params.append([
            f"  {recording_type} Config",
            f"Length: {config['segment_length']}s, Frames: {config['expected_frames']}"
        ])
    
    print(tabulate(preprocessing_params, headers=["Parameter", "Value"], tablefmt="grid"))

def print_model_architecture(model):
    print("\n=== Model Architecture Parameters ===")
    model_params = [
        ["Backbone", "EfficientNet-B2"],
        ["Backbone Channels", "1408"],
        ["Attention Heads", "4"],
        ["Backbone Dropout", "0.3"],
        ["Path Dropout", "0.2"],
        ["Spatial Dropout", "0.3"],
        ["Classifier Hidden Size", "512"],
        ["First Classifier Dropout", "0.5"],
        ["Subsequent Classifier Dropout", "0.3"],
        ["Total Parameters", f"{sum(p.numel() for p in model.parameters()):,}"],
        ["Trainable Parameters", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"]
    ]
    print(tabulate(model_params, headers=["Parameter", "Value"], tablefmt="grid"))

def print_dataset_info(dataset, loader):
    print("\n=== Dataset Configuration ===")
    dataset_info = [
        ["Total Samples", len(dataset)],
        ["Batch Size", loader.batch_size],
        ["Number of Batches", len(loader)],
        ["Shuffle", "False"],
        ["Number of Workers", 0]
    ]
    print(tabulate(dataset_info, headers=["Parameter", "Value"], tablefmt="grid"))

def evaluate_saved_model():
    # 1. Print System Information
    print_system_info()
    
    # 2. Initialize model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PSLAModel(num_classes=2).to(device)
    
    # 3. Print Model Architecture
    print_model_architecture(model)
    
    # 4. Load weights
    print("\nLoading model weights from 'best_model.pth'...")
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    # 5. Initialize preprocessor and print parameters
    preprocessor = AudioPreprocessor()
    print_preprocessing_params(preprocessor)
    
    # 6. Load dataset
    print("\nPreparing dataset...")
    dataset = ParkinsonsDataset(
        sick_dir='./sick/',
        healthy_dir='./healthy/'
    )
    dataset.set_training(False)
    
    # 7. Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=custom_collate
    )
    
    # 8. Print Dataset Information
    print_dataset_info(dataset, loader)
    
    # 9. Evaluate
    print("\nRunning evaluation...")
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for specs, labels in tqdm(loader, desc="Processing batches"):
            specs = specs.to(device)
            outputs = model(specs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 10. Generate and save all visualizations
    print("\nGenerating visualizations...")
    
    # 10.1 Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'PD'],
                yticklabels=['Healthy', 'PD'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 10.2 ROC Curve
    plot_roc_curve(all_labels, all_probs)
    
    # 10.3 Precision-Recall Curve
    plot_precision_recall_curve(all_labels, all_probs)
    
    # 10.4 Confidence Distribution
    plot_confidence_distribution(all_probs, all_labels)
    
    # 11. Print detailed metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Healthy', 'PD']))
    
    # 12. Analyze prediction confidence
    confidence_healthy = all_probs[all_labels == 0]
    confidence_pd = all_probs[all_labels == 1]
    
    print("\nConfidence Analysis:")
    print(f"Healthy predictions - Mean: {confidence_healthy.mean():.3f} ± {confidence_healthy.std():.3f}")
    print(f"PD predictions - Mean: {confidence_pd.mean():.3f} ± {confidence_pd.std():.3f}")
    
    # 13. Analyze misclassifications
    misclassified = all_labels != all_preds
    print(f"\nMisclassified samples: {misclassified.sum()} ({misclassified.sum()/len(all_labels)*100:.2f}%)")
    
    # 14. Additional Performance Metrics
    print("\n=== Detailed Performance Metrics ===")
    performance_metrics = [
        ["Metric", "Value"],
        ["Accuracy", f"{(all_preds == all_labels).mean():.3f}"],
        ["ROC AUC", f"{roc_auc_score(all_labels, all_probs):.3f}"],
        ["Misclassification Rate", f"{(all_preds != all_labels).mean():.3f}"],
        ["Total Samples", len(all_labels)],
        ["Misclassified Samples", f"{(all_preds != all_labels).sum()} ({(all_preds != all_labels).sum()/len(all_labels)*100:.2f}%)"]
    ]
    print(tabulate(performance_metrics, headers="firstrow", tablefmt="grid"))
    
    # Class-specific metrics
    print("\n=== Class-Specific Metrics ===")
    class_metrics = [
        ["Metric", "Healthy (0)", "PD (1)"],
        ["Samples", f"{(all_labels == 0).sum()}", f"{(all_labels == 1).sum()}"],
        ["Mean Confidence", f"{all_probs[all_labels == 0].mean():.3f}", f"{all_probs[all_labels == 1].mean():.3f}"],
        ["Confidence Std", f"{all_probs[all_labels == 0].std():.3f}", f"{all_probs[all_labels == 1].std():.3f}"]
    ]
    print(tabulate(class_metrics, headers="firstrow", tablefmt="grid"))
    
    print("\nEvaluation complete! Check the generated visualization files:")
    print("1. confusion_matrix.png")
    print("2. roc_curve.png")
    print("3. pr_curve.png")
    print("4. confidence_dist.png")
    
    return all_labels, all_preds, all_probs

if __name__ == '__main__':
    labels, preds, probs = evaluate_saved_model()