import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms
import PIL
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm 
from torch.utils.data import DataLoader


from models import create_linear_input, LinearClassifier, AttentiveClassifier
from datasets.datasets import EmbeddingsDataset

def get_transformation(model):
    if model == 'dinov2' or model == 'dinov2_reg':
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.Resampling.BILINEAR, antialias=True),
            transforms.ToTensor()
        ])
    elif model == "megadescriptor":
        return transforms.Compose([
        transforms.Resize((384, 384), interpolation= PIL.Image.Resampling.BILINEAR, antialias=True),
        transforms.ToTensor(),
    ])
    else:
        raise ValueError(f"Unknown model: {model}")

def compute_full_embeddings(dataloader, feature_model, device):
    embeddings = []
    labels = []
    feature_model.eval()
    with torch.no_grad():
        for images, lbls in tqdm(dataloader):
            images = images.unsqueeze(0).to(device)
            features = feature_model(images)
            ((patch_tokens, class_token),) = features
            patch_tokens = patch_tokens.squeeze(0)  # should remove leading 1
            class_token = class_token.squeeze(0)    # should remove leading 1
            embeddings.append((patch_tokens.clone(), class_token.clone()))
            labels.append(torch.tensor(lbls))
    return embeddings, labels

def compute_embeddings(dataloaders, feature_model, device):
    embeddings = []
    labels = []
    for loader in dataloaders:
        e, l = compute_full_embeddings(loader, feature_model, device)
        embeddings.append(e)
        labels.append(l)
    return embeddings, labels

def flatten_embeddings(embeddings, labels, pooling_method, use_class, attentive_classifier=None):
    embeddings_f = []

    if pooling_method == 'attentive':
        for embedding in embeddings:
            device = embedding[0].device
            attentive_classifier = attentive_classifier.to(device)
            attended_output = attentive_classifier.get_pooled_output(embedding)
            embeddings_f.append(attended_output.cpu().detach().numpy())
    else:
        for embedding in embeddings:
            linear_input = create_linear_input(embedding, use_avgpool=(pooling_method == 'linear'), use_class=use_class)
            embeddings_f.extend([np.array(tr.cpu()) for tr in linear_input])

    labels_f = [np.array(l.cpu()) for label in labels for l in label]
    embeddings_f = np.vstack(embeddings_f)
    return np.array(embeddings_f), np.array(labels_f)

def combine_train_val(train_embeddings, train_labels, val_embeddings, val_labels):
    combined_embeddings = train_embeddings + val_embeddings
    combined_labels = train_labels + val_labels
    return combined_embeddings, combined_labels

def train_linear_classifier(train_embeddings, train_labels, use_class, use_avgpool, device, num_classes=1000, num_epochs=10, learning_rate=5e-3):
    # Create the embeddings dataset and dataloader
    train_dataset = EmbeddingsDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False)

    classifier = LinearClassifier(num_classes=num_classes, use_class=use_class, use_avgpool=use_avgpool).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        classifier.train()
        total_loss = 0.0
        for patch_tokens, class_token, labels in train_loader:
            patch_tokens = patch_tokens.to(device).float()  
            class_token = class_token.to(device).float()    
            labels = labels.to(device).long()              
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = classifier((patch_tokens, class_token))
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return classifier

def train_attentive_classifier(train_embeddings, train_labels, use_class, device, num_classes=1000, num_epochs=10, learning_rate=1e-5, complete_block=False):
    # Create the embeddings dataset and dataloader
    train_dataset = EmbeddingsDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False)

    attentive_classifier = AttentiveClassifier(num_classes=num_classes, use_class=use_class, complete_block=complete_block).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(attentive_classifier.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        attentive_classifier.train()
        total_loss = 0.0
        for patch_tokens, class_token, labels in train_loader:
            patch_tokens = patch_tokens.to(device).float()  
            class_token = class_token.to(device).float()    
            labels = labels.to(device).long()              
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = attentive_classifier((patch_tokens, class_token))
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return attentive_classifier

def train_val_linear_classifier(train_embeddings, train_labels, val_embeddings, val_labels, use_class, use_avgpool, device, num_classes=1000, num_epochs=50, learning_rate=5e-3, batch_size=32, patience=5):
    # Create the embeddings dataset and dataloader
    train_dataset = EmbeddingsDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = EmbeddingsDataset(val_embeddings, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    classifier = LinearClassifier(num_classes=num_classes, use_class=use_class, use_avgpool=use_avgpool).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    best_val_acc = 0.0  # Track the best validation accuracy
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        if early_stop:
            break
        
        classifier.train()
        total_loss = 0.0
        for patch_tokens, class_token, labels in train_loader:
            patch_tokens = patch_tokens.to(device).float()  
            class_token = class_token.to(device).float()    
            labels = labels.to(device).long()              
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = classifier((patch_tokens, class_token))
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Evaluate on validation set
        classifier.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for patch_tokens, class_token, labels in val_loader:
                patch_tokens = patch_tokens.to(device).float()
                class_token = class_token.to(device).float()
                labels = labels.to(device).long()

                outputs = classifier((patch_tokens, class_token))
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        
        # Track the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Early stopping logic based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1  # Save the best epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                early_stop = True
    
    # Return the epoch with the lowest validation loss and the highest validation accuracy
    return best_epoch, best_val_acc

def train_val_attentive_classifier(train_embeddings, train_labels, val_embeddings, val_labels, use_class, device, num_classes=1000, num_epochs=50, learning_rate=1e-5, batch_size=32, patience=5, complete_block=False):
    # Create the embeddings dataset and dataloader
    train_dataset = EmbeddingsDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False)

    val_dataset = EmbeddingsDataset(val_embeddings, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)

    attentive_classifier = AttentiveClassifier(num_classes=num_classes, use_class=use_class, complete_block=complete_block).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(attentive_classifier.parameters(), lr=learning_rate)
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    best_val_acc = 0.0  # Track the best validation accuracy
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        if early_stop:
            break
        
        attentive_classifier.train()
        total_loss = 0.0
        for patch_tokens, class_token, labels in train_loader:
            patch_tokens = patch_tokens.to(device).float()  
            class_token = class_token.to(device).float()    
            labels = labels.to(device).long()              
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = attentive_classifier((patch_tokens, class_token))
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        # Evaluate on validation set
        attentive_classifier.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for patch_tokens, class_token, labels in val_loader:
                patch_tokens = patch_tokens.to(device).float()
                class_token = class_token.to(device).float()
                labels = labels.to(device).long()

                outputs = attentive_classifier((patch_tokens, class_token))
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        
        # Track the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Early stopping logic based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1  # Save the best epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                early_stop = True
    
    # Return the epoch with the lowest validation loss and the highest validation accuracy
    return best_epoch, best_val_acc


def compute_scores(outputs):
    softmax_scores = F.softmax(outputs, dim=1)
    max_softmax_scores, _ = torch.max(softmax_scores, dim=1)
    max_logit_scores, _ = torch.max(outputs, dim=1)
    
    return max_softmax_scores, max_logit_scores

def compute_top1_accuracy(outputs, labels):
    # Ensure outputs and labels are tensors and on the same device
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
        
    # Ensure outputs and labels are on the same device
    labels = labels.to(outputs.device)

    _, predicted = torch.max(outputs, 1)

    if predicted.shape != labels.shape:
        raise ValueError(f"Shape mismatch: predicted shape {predicted.shape}, labels shape {labels.shape}")
    
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    
    return accuracy

def eval_closed_set(embeddings, labels, model, device='cuda'):
    model.eval()

    test_dataset = EmbeddingsDataset(embeddings, labels)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)
    outputs = []
    with torch.no_grad():
        for patch_tokens, class_token, _ in test_loader:          
            outputs.append(model((patch_tokens, class_token)))
    
    outputs = torch.cat(outputs, dim=0)
    accuracy = compute_top1_accuracy(outputs, labels)
    max_softmax_scores, max_logit_scores = compute_scores(outputs)

    return accuracy, max_softmax_scores, max_logit_scores

def eval_open_set(embeddings, labels, model, device='cuda'):
    model.eval()

    test_dataset = EmbeddingsDataset(embeddings, labels)
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=False)
    outputs = []
    with torch.no_grad():
        for patch_tokens, class_token, _ in test_loader:            
            outputs.append(model((patch_tokens, class_token)))
    
    outputs = torch.cat(outputs, dim=0)
    max_softmax_scores, max_logit_scores = compute_scores(outputs)

    return max_softmax_scores, max_logit_scores

def evaluate_knn(train_embeddings, test_embeddings, train_labels, test_labels):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(train_embeddings)
    test_knn_distances, test_knn_indices = neighbors.kneighbors(test_embeddings)
    t1_hits = sum(1 for i, label in enumerate(test_labels) if train_labels[test_knn_indices[i][0]] == label)
    top1_accuracy = t1_hits / len(test_labels)

    min_distances = [elem[0] for elem in test_knn_distances]
    return min_distances, top1_accuracy

def get_ROC(closed_metric, open_metric, plot=False, knn=True):
    if isinstance(closed_metric, np.ndarray) or isinstance(closed_metric, torch.Tensor):
        closed_metric = closed_metric.tolist()
    if isinstance(open_metric, np.ndarray) or isinstance(open_metric, torch.Tensor):
        open_metric = open_metric.tolist()
    if knn:
        yt = [1 for _ in open_metric] + [0 for _ in closed_metric]
    else:
        yt = [0 for _ in open_metric] + [1 for _ in closed_metric]
    ys = open_metric + closed_metric
    fpr, tpr, _ = roc_curve(yt, ys)

    if plot:
        plt.rcParams["figure.figsize"] = (8,8)
        plt.rcParams.update({'font.size': 22})
        label_str = f'(AUROC: {auc(fpr, tpr):.3f})'
        plt.plot(fpr, tpr, label=label_str)

        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([-0.01, 1.0])
        plt.ylim([-0.01, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc='lower right')
        plt.title('Closed/Open Classification ROC')
        plt.show()
        # plt.savefig('roc.png')

    return auc(fpr, tpr)
