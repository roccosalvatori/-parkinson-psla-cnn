import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
import numpy as np
from model import PSLAModel
from preprocessing import ParkinsonsDataset, custom_collate
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
from tabulate import tabulate

def get_color_palette(base_color='001631', n_colors=4):
    """Generate shades of the base color"""
    import colorsys
    # Convert hex to RGB
    rgb = tuple(int(base_color[i:i+2], 16)/255 for i in (0, 2, 4))
    # Convert to HSV
    hsv = colorsys.rgb_to_hsv(*rgb)
    colors = []
    for i in range(n_colors):
        # Vary saturation and value while keeping hue constant
        sat = min(1.0, hsv[1] * (1.0 - i * 0.2))
        val = min(1.0, hsv[2] * (1.0 + i * 0.2))
        rgb = colorsys.hsv_to_rgb(hsv[0], sat, val)
        colors.append(rgb)
    return colors

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=50, device='cuda', early_stopping_patience=10):
    best_val_acc = 0.0
    no_improve_count = 0
    
    # Lists to store metrics for plotting
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Visualize first batch before training
    first_batch, first_labels = next(iter(train_loader))
    print("\nFirst batch shape:", first_batch.shape)
    print("First batch labels:", first_labels)
    
    # Save first mel spectrogram
    first_mel = first_batch[0].squeeze()  # Remove batch and channel dimensions
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        first_mel.numpy(),  # Convert to numpy array
        sr=16000,
        hop_length=160,
        x_axis='time',
        y_axis='mel',
        fmax=8000,
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'First Sample (Label: {"Sick" if first_labels[0] == 1 else "Healthy"})')
    plt.savefig('first_sample_mel.png')
    plt.close()
    
    print("\nMel Spectrogram Details:")
    print(f"Shape: {first_mel.shape}")
    print(f"Value range: [{first_mel.min():.2f}, {first_mel.max():.2f}]")
    print(f"Mean: {first_mel.mean():.2f}")
    print(f"Std: {first_mel.std():.2f}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_idx, (specs, labels) in enumerate(train_loader):
            # specs shape: [batch_size, channels, height, width]
            specs, labels = specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print metrics in table format
        metrics_table = [
            ["Metric", "Training", "Validation"],
            ["Loss", f"{train_loss:.4f}", f"{val_loss:.4f}"],
            ["Accuracy", f"{train_acc:.4f}", f"{val_acc:.4f}"]
        ]
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(tabulate(metrics_table, headers="firstrow", tablefmt="grid"))
        
        # Update scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_count += 1
            if no_improve_count >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Plot training progress
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            plt.figure(figsize=(15, 5))
            
            # Plot losses
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.title('Loss over epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot accuracies
            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Train Acc')
            plt.plot(val_accs, label='Val Acc')
            plt.title('Accuracy over epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_progress.png')
            plt.close()

    return train_losses, train_accs, val_losses, val_accs

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training history with custom style"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
    fig.patch.set_facecolor('white')
    
    # Get epochs range
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, '-', label='Training Loss', color='#2E86C1', linewidth=2)
    ax1.plot(epochs, val_losses, '-', label='Validation Loss', color='#E74C3C', linewidth=2)
    ax1.set_title('Loss Over Time', pad=15, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(frameon=True, facecolor='white', framealpha=1)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_facecolor('#f8f9fa')
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, '-', label='Training Accuracy', color='#27AE60', linewidth=2)
    ax2.plot(epochs, val_accs, '-', label='Validation Accuracy', color='#8E44AD', linewidth=2)
    ax2.set_title('Accuracy Over Time', pad=15, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(frameon=True, facecolor='white', framealpha=1)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_facecolor('#f8f9fa')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('training_history.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    """Plot confusion matrix with custom style"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and get figure object
    fig = plt.figure(figsize=(8, 6), dpi=100)
    fig.patch.set_facecolor('white')  # Set figure background color
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 12})
    
    plt.title(title, pad=15, fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_prob, title='ROC Curve', save_path=None):
    """Plot ROC curve with custom style"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6), dpi=100)
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    
    plt.plot(fpr, tpr, color='#2E86C1', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='#7F8C8D', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title(title, pad=15, fontsize=12, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, facecolor='white', framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_model(model, loader, device):
    """Comprehensive model evaluation"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for specs, labels in loader:
            # Handle 4D tensor [batch_size, channels, height, width]
            if len(specs.shape) == 5:  # If 5D tensor [batch, segments, channels, height, width]
                B, S, C, H, W = specs.shape
                specs = specs.view(-1, C, H, W)  # Reshape to 4D
            
            specs = specs.to(device)
            outputs = model(specs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset with both structures
    dataset = ParkinsonsDataset(
        sick_dir='./sick/',
        healthy_dir='./healthy/',
        sick_dir2='./sick2/',
        healthy_dir2='./healthy2/'
    )
    
    # Print dataset statistics
    dataset.print_dataset_stats()
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate,
        pin_memory=True
    )
    
    # Initialize model, criterion, optimizer, scheduler
    model = PSLAModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=0.0003,
                           weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # During training
    dataset.set_training(True)
    
    # Train model with scheduler
    train_losses, train_accs, val_losses, val_accs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device,
        early_stopping_patience=10
    )
    
    # During validation/testing
    dataset.set_training(False)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on training set
    print("\nEvaluating on training set:")
    train_labels, train_preds, train_probs = evaluate_model(model, train_loader, device)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set:")
    val_labels, val_preds, val_probs = evaluate_model(model, val_loader, device)
    
    # Plot all metrics
    print("\nPlotting metrics...")
    
    # Training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # Confusion matrices
    print("\nTraining Set Metrics:")
    print(classification_report(train_labels, train_preds))
    plot_confusion_matrix(train_labels, train_preds, 
                         title='Training Confusion Matrix',
                         save_path='train_confusion_matrix.png')
    
    print("\nValidation Set Metrics:")
    print(classification_report(val_labels, val_preds))
    plot_confusion_matrix(val_labels, val_preds, 
                         title='Validation Confusion Matrix',
                         save_path='val_confusion_matrix.png')
    
    # ROC curves
    plot_roc_curve(train_labels, train_probs, 
                   title='Training ROC Curve',
                   save_path='train_roc_curve.png')
    plot_roc_curve(val_labels, val_probs, 
                   title='Validation ROC Curve',
                   save_path='val_roc_curve.png')
    
    print("\nAll plots have been saved:")
    print("- training_history.png")
    print("- train_confusion_matrix.png")
    print("- val_confusion_matrix.png")
    print("- train_roc_curve.png")
    print("- val_roc_curve.png")

    # Print dataset summary
    dataset_summary = [
        ["Split", "Size", "Batch Size", "Batches"],
        ["Training", len(train_dataset), 8, len(train_loader)],
        ["Validation", len(val_dataset), 8, len(val_loader)]
    ]
    print("\nDataset Summary:")
    print(tabulate(dataset_summary, headers="firstrow", tablefmt="grid"))

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_summary = [
        ["Parameter", "Value"],
        ["Total Parameters", f"{total_params:,}"],
        ["Trainable Parameters", f"{trainable_params:,}"],
        ["Learning Rate", f"{optimizer.param_groups[0]['lr']:.6f}"],
        ["Device", device]
    ]
    print("\nModel Summary:")
    print(tabulate(model_summary, headers="firstrow", tablefmt="grid"))

if __name__ == '__main__':
    main() 