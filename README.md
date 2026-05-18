# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset

### Problem Statement
To develop an image classification model using **Transfer Learning** with the **VGG19 architecture** for classifying images from the given dataset.

### Dataset
The dataset consists of image samples organized into **training** and **testing** folders with different class categories.

<img width="438" height="182" alt="image" src="https://github.com/user-attachments/assets/7d26d902-7ae5-436d-9487-c2c1576e2a3c" />

---

## Neural Network Model
<img width="715" height="517" alt="image" src="https://github.com/user-attachments/assets/99335698-c3c4-4c5d-95ee-e310d2f85ad4" />


---

## DESIGN STEPS

### STEP 1:
Load and preprocess the dataset by resizing images and converting them into tensors.

### STEP 2:
Load the pre-trained **VGG19 model** and modify the final fully connected layer according to the dataset classes.

### STEP 3:
Freeze the feature extraction layers and define the loss function and optimizer.

### STEP 4:
Train the model using the training dataset and validate using the testing dataset.

### STEP 5:
Evaluate the model performance using confusion matrix and classification report.

### STEP 6:
Predict the output for new sample images and compare actual and predicted results.

---

## PROGRAM

### Name: THARUN D

### Register Number: 212223240167

```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes
model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features,
    1
)

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Train the model
train_model(
    model,
    train_loader,
    test_loader
)
```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot
<img width="761" height="546" alt="image" src="https://github.com/user-attachments/assets/8533bf94-0f2b-477e-8179-3a4ea763bbe3" />


## Confusion Matrix
<img width="696" height="499" alt="Screenshot 2026-05-18 134712" src="https://github.com/user-attachments/assets/b9fbde0a-191b-4e86-b7b3-e7c31c9a9074" />


## Classification Report
<img width="450" height="181" alt="Screenshot 2026-05-18 134729" src="https://github.com/user-attachments/assets/81ab6121-4314-4399-a7bc-0449653d07cc" />


### New Sample Data Prediction
<img width="466" height="452" alt="image" src="https://github.com/user-attachments/assets/175ca8a7-16c2-44f6-aa66-0dc61eb22ec4" />


## RESULT
Thus, an image classification model using **Transfer Learning with VGG19 architecture** was successfully developed and trained for the given dataset.
