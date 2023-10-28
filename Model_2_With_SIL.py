from transformers import AutoModelForSequenceClassification
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.nn.functional import softmax
from Config_Manager import get_dataset, SEED, LEARNING_RATE, BATCH_SIZE, DEVICE, GENERATIONS, STUDENT_EPOCHS, TEACHER_EPOCHS
import sys

"""
HYPER PARAMS FROM CONFIG FILE
"""
seed = SEED
learning_rate = LEARNING_RATE
batch_size = BATCH_SIZE
device = DEVICE
generations = GENERATIONS
student_epochs = STUDENT_EPOCHS
teacher_epochs = TEACHER_EPOCHS

def custom_loss(predictions, labels):
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(predictions, labels)


#get data sets and splits
dataset = get_dataset("masakhane")
train_dataset = dataset["train"]
val_dataset = dataset["val"]
val_dataset_d1 = get_dataset("naija")["val"]
del dataset

#create data loaders, create a vaidation data loader for data sets 1 and 2
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
val_dataloader_d1 = DataLoader(val_dataset_d1, batch_size=batch_size)

#Make the 2 models on the afriberta pre trained model from step 1
student_model = AutoModelForSequenceClassification.from_pretrained("Saved_Models/model_1").to(device)
student_model.config.loss_name = "cross_entropy" #use cross entropy loss function
student_optimizer = AdamW(student_model.parameters(), lr=learning_rate)

student_model.train()

num_training_steps = student_epochs * len(train_dataloader)
training_steps = student_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=student_optimizer,
    num_training_steps=training_steps,
    num_warmup_steps=0
)


prog_bar = tqdm(range(training_steps))

train_epoch_loss = []
val_epoch_loss = []
val_epoch_loss_d1 = []

for gen in range(generations):
    teacher_model = AutoModelForSequenceClassification.from_pretrained("Saved_Models/model_1").to(device)
    teacher_model.config.loss_name = "cross_entropy" #use cross entropy loss function

    #copy the student model to be the teacher model
    teacher_model.load_state_dict(student_model.state_dict())
    
    #define optimizer & scheduler for teacher model
    teacher_optimizer = AdamW(teacher_model.parameters(), lr=learning_rate)
    teacher_optimizer.zero_grad()
    training_steps_1 = teacher_epochs * len(train_dataloader)
    lr_scheduler_1 = get_scheduler(
        name="linear",
        optimizer=teacher_optimizer,
        num_training_steps=training_steps_1,
        num_warmup_steps=0
    )

    new_batch = []
    for te in range(teacher_epochs):
        new_batch = [] #Empty it so we only take the last set of pseudo labels from the last epoch 
        for batch in train_dataloader:
            #First train the teacher model 
            batch = {k : v.to(device) for k, v in batch.items()}
            teacher_model.train()
            outputs = teacher_model(**batch) #get outputs from teacher model
                
            loss = outputs.loss
            loss.backward()
            teacher_optimizer.step()
            lr_scheduler_1.step()
            teacher_optimizer.zero_grad()
            prog_bar.update(1)

            # Get teacher model predictions for the inputs next (new labels)

            with torch.no_grad():
                teacher_logits = teacher_model(**batch).logits

            # softmax the teacher logits for pseudo-labels
            pseudo_labels = softmax(teacher_logits, dim=1) 

            temp_batch = batch.copy()
            temp_batch["labels"] = pseudo_labels
            new_batch.append(temp_batch)

    se_loss = []
    for se in range(student_epochs):
        student_model.train()
        step_loss = []
        for batch in new_batch:
            batch = {k : v.to(device) for k, v in batch.items()}
            # Train the student model using the teacher pseudo-labels (soft labels)
            student_optimizer.zero_grad()
            student_logits = student_model(**batch).logits
            loss = custom_loss(student_logits, batch["labels"])
            loss.backward()
            student_optimizer.step()
            lr_scheduler.step()
            step_loss.append(loss.item())
        se_loss.append(np.mean(step_loss)) #keep track of loss
    
    #Loss for 1 generation averaged over the n student epochs 
    train_epoch_loss.append(np.mean(se_loss))
        
    
    #Evaluate the model on the validation set for dataset 2
    student_model.eval()
    step_loss = []
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = student_model(**batch)
        loss = outputs.loss
        step_loss.append(loss.item())
    
    val_epoch_loss.append(np.mean(step_loss))

    #Evaluate the model on the validation set for dataset 1
    student_model.eval()
    step_loss = []
    for batch in val_dataloader_d1:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = student_model(**batch)
        loss = outputs.loss
        step_loss.append(loss.item())
    
    val_epoch_loss_d1.append(np.mean(step_loss))
    

loss_data = [
    train_epoch_loss,
    val_epoch_loss,
    val_epoch_loss_d1
]

loss_data = np.array(loss_data)

#Save the student model 
student_model.save_pretrained("Saved_Models/model_2_SIL")

np.save(f"Saved_Models/model_2_SIL/model_2_SIL_Loss_{sys.argv[1]}.npy", loss_data)



