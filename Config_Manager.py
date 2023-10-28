#IMPORTS
from transformers import AutoTokenizer
import pandas as pd
from  datasets  import  load_dataset
import numpy as np
import evaluate
import torch
import evaluate
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

"""
HYPER PARAMS FOR MODELS
=======================
"""
SEED = 42
CLASSES = 10
EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 16
GENERATIONS = 3
STUDENT_EPOCHS = 3
TEACHER_EPOCHS = 3
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
"""
=======================
"""


"""
Tokenize function, tokenizes the string data in the "text" field of the dataset
"""
def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("castorini/afriberta_base", use_fast=False)
    tokenizer.model_max_length = 512
    return tokenizer(examples["text"], padding="max_length", truncation=True)

"""
Function to edit the labels of the various datasets to have the same number of output classes
i.e. because we have 3 classes from the first dataset and 7 labels in the second, both labels should be 
edited to contain 10 classes 
"""
def one_hot_encode_labels(dataset, dataset_number, extend_classes = 10):
    orig_labels = np.array(dataset["labels"])
    new_labels = torch.zeros(extend_classes)

    if dataset_number == 1: #If were in dataset 1 , we need to add 7 zeros (classes) to the end
        new_labels[orig_labels] = 1 #Set the first 3 elements to the dataset 1 labels
    else: #If we're in dataset 2. we need to add 3 zeros (classes) to the beginning 
        new_labels[orig_labels + 3] = 1 #Set the last 7 elements to the dataset 2 labels
    
    #Return the original text with new one hot encoded labels
    return {
        "text" : dataset["text"],
        "labels" : new_labels
    }

"""
Function to map the original labels to the newly created one-hot labels
where the newly created one-hot lables have the length equal to the total number of classes
here: 3 classes from data set 1 + 7 classes from data set 2 = 10 classes in total
"""
def construct_labels(dataset, dataset_number):
    return dataset.map(lambda datapoint : one_hot_encode_labels(datapoint, dataset_number))


"""
A funtion to get the tokenized formatted dataset.
PARAMS : dataset - the dataset to be tokenized (either naija or masakhane)
            seed - the seed to be used for shuffling the dataset
RETURN : a dictionary containing the train, test and validation sets
"""
def get_dataset(dataset, seed = 42):
    tokenized_datasets = None
    if dataset == "naija":
        #First dataset is Igbo from the Naija Sentimient Twitter Dataset
        ds = load_dataset("HausaNLP/NaijaSenti-Twitter", "ibo")

        #Fix the column names to work with our functions
        ds = ds.rename_column("label", "labels") 
        ds = ds.rename_column("tweet", "text")

        # Tokenize the data set
        tokenized_datasets = construct_labels(ds, 1) 

    else:
        #Second dataset is Hausa from Masakhanews text classification dataset
        ds = load_dataset('masakhane/masakhanews', 'hau')

        #Fix the column names to work with our functions
        ds = ds.remove_columns(["text", "url", "headline"])
        ds = ds.rename_column("label", "labels")
        ds = ds.rename_column("headline_text", "text")

        #tokenize the data set
        tokenized_datasets = construct_labels(ds, 2)

    del ds

    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    #Create train, validation and testing sets 
    train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
    test_dataset = tokenized_datasets["test"].shuffle(seed=seed)
    val_dataset = tokenized_datasets["validation"].shuffle(seed=seed)

    del tokenized_datasets

    return {
        "train" : train_dataset,
        "test" : test_dataset,
        "val" : val_dataset
    }


"""
A function to compute the metrics of the model
PARAMS : logits - the logits of the model
         labels - the labels of the model
RETURN : a dictionary containing the metrics of the model
"""
def compute_metrics(logits, labels):
    # logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels=np.argmax(labels, axis=-1)


    # Use evaluate.load to load pre-defined accuracy
    acc = evaluate.load("accuracy")
    accuracy = acc.compute(predictions=predictions, references=labels)

    # Use evaluate.load to load pre-defined F1, precision, and recall metrics
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    f1_value = f1.compute(predictions=predictions, references=labels, average='weighted')  # You can adjust 'average' if needed
    precision_value = precision.compute(predictions=predictions, references=labels, average='weighted')  # You can adjust 'average' if needed
    recall_value = recall.compute(predictions=predictions, references=labels, average='weighted', zero_division=0.0)  # You can adjust 'average' if needed

    cohens_kappa=cohen_kappa_score(labels, predictions)

    conf_matrx = confusion_matrix(labels, predictions, labels=np.arange(CLASSES))

    return {
        "accuracy": accuracy['accuracy'],
        "f1": f1_value['f1'],
        "precision": precision_value['precision'],
        "recall": recall_value['recall'],
        "cohenkappa": cohens_kappa,
        "confusion_matrix": conf_matrx
    }


"""
A function to evaluate the model on the provided dataloader, uses the custom metrics from above 
PARAMS : model - the model to be evaluated
         dataloader - the dataloader to be used for evaluation
RETURN : a dictionary containing the metrics of the model
"""
def evaluate_model(model, dataloader):
    model.eval()
    true_labels = []
    predicted_labels = []
    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        true = batch["labels"]

        true_labels.extend(true.cpu().numpy())
        predicted_labels.extend(logits.cpu().numpy())

    custom_metrics_dict = compute_metrics(np.array(predicted_labels), np.array(true_labels))
    return custom_metrics_dict

"""
A function to save the confusion matrix
PARAMS : cm - the confusion matrix to be saved
         name - the name of the confusion matrix
"""
def save_cm(cm, name, title):
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', cbar=False, annot_kws={'size': 14}, square=True)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.title(f'{title}', fontsize=16)
    plt.savefig(f'Plots/{name}.png', dpi=300)
    plt.close()

"""
A function to plot the loss curves of model 1
PARAMS : runs - the number of runs to plot
"""
def plot_model_1_loss(runs = 2):
    val_loss = []
    train_loss = []
    for i in range(runs):
        model_1_loss = np.load(f"Saved_Models/model_1/Model_1_Loss_{i + 1}.npy")
        train_loss.append(model_1_loss[0])
        val_loss.append(model_1_loss[1])
    
    train_loss_mean = np.mean(np.array(train_loss), axis=0)
    val_loss_mean = np.mean(np.array(val_loss), axis=0)
    train_loss_var = np.var(np.array(train_loss), axis=0)
    val_loss_var = np.var(np.array(val_loss), axis=0)

    plt.plot(train_loss_mean, label='Training Loss')
    plt.plot(val_loss_mean,label='Validation Loss')

    plt.fill_between(np.arange(0, len(train_loss_mean), 1), train_loss_mean - train_loss_var, train_loss_mean + train_loss_var, alpha=0.2, color='r')
    plt.fill_between(np.arange(0, len(val_loss_mean), 1), val_loss_mean - val_loss_var, val_loss_mean + val_loss_var, alpha=0.2, color='g')

    plt.legend()
    plt.xticks(np.arange(0, len(train_loss_mean), 1))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.title("Model 1 Loss on Dataset 1 (NaijaSenti)")
    plt.savefig("Plots/Model_1_Loss.png", dpi = 300)
    plt.close()


"""
A function to plot the loss curves of model 2 with and witout SIL
PARAMS : SIL - whether to plot the model with or without SIL
         runs - the number of runs to plot
"""
def plot_model_2_loss(runs = 2, SIL = True):
    val_loss_d1 = []
    val_loss_d2 = []
    train_loss = []
    
    for i in range(runs):
        if SIL :
            model_1_loss = np.load(f"Saved_Models/model_2_SIL/Model_2_SIL_Loss_{i + 1}.npy")
        else:
            model_1_loss = np.load(f"Saved_Models/model_2_No_SIL/Model_2_No_SIL_Loss_{i + 1}.npy")

        train_loss.append(model_1_loss[0])
        val_loss_d1.append(model_1_loss[2])
        val_loss_d2.append(model_1_loss[1])
    
    train_loss_mean = np.mean(np.array(train_loss), axis=0)
    val_loss_mean_d1 = np.mean(np.array(val_loss_d1), axis=0)
    val_loss_mean_d2 = np.mean(np.array(val_loss_d2), axis=0)

    train_loss_var = np.var(np.array(train_loss), axis=0)
    val_loss_var_d1 = np.var(np.array(val_loss_d1), axis=0)
    val_loss_var_d2 = np.var(np.array(val_loss_d2), axis=0)
    

    plt.plot(train_loss_mean, label='Training Loss', color = 'r')
    plt.plot(val_loss_mean_d1, label='Validation Loss Dataset 1', color = 'g')
    plt.plot(val_loss_mean_d2, label='Validation Loss Dataset 2', color = 'b')

    plt.fill_between(np.arange(0, len(train_loss_mean), 1), train_loss_mean - train_loss_var, train_loss_mean + train_loss_var, alpha=0.2, color='r')
    plt.fill_between(np.arange(0, len(val_loss_mean_d1), 1), val_loss_mean_d1 - val_loss_var_d1, val_loss_mean_d1 + val_loss_var_d1, alpha=0.2, color='g')
    plt.fill_between(np.arange(0, len(val_loss_mean_d2), 1), val_loss_mean_d2 - val_loss_var_d2, val_loss_mean_d2 + val_loss_var_d2, alpha=0.2, color='b')

    plt.legend()
    plt.xticks(np.arange(0, len(train_loss_mean), 1))
    plt.ylim(0, 1)
    
    plt.ylabel("Loss")

    if SIL:
        plt.xlabel("Generation")
        plt.title("Model 2 With SIL Loss on Dataset 1 & 2")
        plt.savefig("Plots/Model_2_SIL_Loss", dpi = 300)
    else:
        plt.xlabel("Epoch")
        plt.title("Model 2 Without SIL Loss on Dataset 1 & 2")
        plt.savefig("Plots/Model_2_No_SIL_Loss", dpi = 300)
    
    plt.close()
        
    