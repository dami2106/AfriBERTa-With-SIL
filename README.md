# NLP Project exploring the effects of SIL 
## Code by Damion, Gavin, Michaela 

## How to run the code:
### Training the models
In order to train the models, 3 model files have been provided. `Model_1.py`, `Model_2_No_SIL.py`, and `Model_2_With_SIL.py`. Each file corresponds to training the specified model. Along with this, when training the model, a number as an argument is required in order to specify the number run that is. This is needed when plotting variance later on. These files can be run as : `python Model_X.py run_number` where `run_number` is an integer. 
Once the models are trained, they will be saved to the `Saved_Models` directory, along with their corresponding training and validation loss. 

### Evaluating the models
An evaluation code file has been provided, `Evaluate_Models.py`. This file will open and test all 3 models and display a report detailing their corresponding metrics on their datasets. This file will also plot the loss functions using the loss saved from above. The generated confusion matrices are also saved as graphics. All plots are saved to the `Plots` directory. 

### Changing Hyper-parameters
The `Config_Manager.py` file contains multiple constants at the top of the file that can be changed. These constants are imported by the models by default, so changing the parameters here has a global affect to all models. These parameters can be seen at the top of the file. 

## File Structure: 
We note that all code discussed above is located in the `Code` folder. 
```
NLP_Project
│   report.pdf
│   README.md
│
└───Code
│   └───Saved_Models
│   └───Plots
│   │
│   │   Config_Manager.py
│   │   Evaluate_Models.py
│   │   Model_1.py
│   │   Model_2_No_SIL.py
│   │   Model_2_With_SIL.py
```

## Packages & Versions Required:  
 - pandas 2.1.1
 - numpy 1.25.2
 - seaborn 0.13.0
 - torch 2.1.0
 - scikit-learn 1.3.1
 - scipy 1.11.3
 - transformers 4.34.0
 - sentencepiece 0.1.99