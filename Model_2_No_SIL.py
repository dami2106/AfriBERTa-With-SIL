#IMPORTS
from transformers import AutoModelForSequenceClassification
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from Config_Manager import get_dataset, SEED, CLASSES, EPOCHS, LEARNING_RATE, BATCH_SIZE, DEVICE
import sys 

"""
HYPER PARAMS FROM CONFIG FILE
"""
seed = SEED
classes = CLASSES
epochs = EPOCHS
learning_rate = LEARNING_RATE
batch_size = BATCH_SIZE
device = DEVICE

dataset = get_dataset("masakhane")
train_dataset = dataset["train"]
val_dataset = dataset["val"]
val_dataset_d1 = get_dataset("naija")["val"]
del dataset

#Load in the model that was saved before
model = AutoModelForSequenceClassification.from_pretrained("Saved_Models/model_1").to(device)
model.config.loss_name = "cross_entropy" #use cross entropy loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)

#create data loaders
#create validation data loader for both data sets 1 and 2
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
val_dataloader_d1 = DataLoader(val_dataset_d1, batch_size=batch_size)

num_training_steps = epochs * len(train_dataloader) #calculate number of training steps

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

train_epoch_loss = []
val_epoch_loss = []
val_epoch_loss_d1 = []

#train model
progress_bar = tqdm(range(num_training_steps))
for epoch in range(epochs):
    step_loss = []
    model.train()
    for batch in train_dataloader: #iterate through training data
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch) #get model outputs
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        step_loss.append(loss.item()) #store loss

        progress_bar.update(1)
    
    train_epoch_loss.append(np.mean(step_loss)) #store training loss, averaged over batch

    #Evaluate the model on the validation set
    model.eval()
    step_loss = []
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        step_loss.append(loss.item()) #store validation loss

    val_epoch_loss.append(np.mean(step_loss)) #append validation loss, averaged over batch

    #calculate validation loss for data set 1 on model 2
    model.eval()
    step_loss = []
    for batch in val_dataloader_d1:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        step_loss.append(loss.item()) #get validation loss

    val_epoch_loss_d1.append(np.mean(step_loss)) #store validation loss, averaged over batch


loss_data = [
    train_epoch_loss,
    val_epoch_loss,
    val_epoch_loss_d1
]

loss_data = np.array(loss_data)

#Save the model to disk
model.save_pretrained("Saved_Models/model_2_No_SIL")

np.save(f"Saved_Models/model_2_No_SIL/Model_2_No_SIL_Loss_{sys.argv[1]}.npy", loss_data)


