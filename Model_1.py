#IMPORTS
from transformers import AutoModelForSequenceClassification
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from Config_Manager import get_dataset, SEED, CLASSES, EPOCHS, LEARNING_RATE, BATCH_SIZE, DEVICE
from matplotlib import pyplot as plt
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


dataset = get_dataset("naija")
train_dataset = dataset["train"]
val_dataset = dataset["val"]
del dataset


"""
Now we need to train model 1 on the first dataset 
"""
model = AutoModelForSequenceClassification.from_pretrained("castorini/afriberta_base", num_labels=classes).to(device)
model.config.loss_name = "cross_entropy" #use cross entropy loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)

#create data loader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
num_training_steps = epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

train_epoch_loss = []
val_epoch_loss = []

#train model
progress_bar = tqdm(range(num_training_steps))
for epoch in range(epochs):
    model.train()
    step_loss = []
    for i,batch in enumerate(train_dataloader): #iterate through data loader
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch) #get outputs of model
        loss = outputs.loss #calculate loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        step_loss.append(loss.item())

        progress_bar.update(1)

    train_epoch_loss.append(np.mean(step_loss)) #store training loss, averaged over batch

    #Validation
    model.eval()
    step_loss = []
    for i,batch in enumerate(val_dataloader): #calculate validation loss
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        step_loss.append(loss.item())

    val_epoch_loss.append(np.mean(step_loss)) #store validation loss, averaged over batch


loss_data = [
    train_epoch_loss,
    val_epoch_loss
]

loss_data = np.array(loss_data)

#Save the model to disk
model.save_pretrained("Saved_Models/model_1")

np.save(f"Saved_Models/model_1/Model_1_Loss_{sys.argv[1]}.npy", loss_data)





