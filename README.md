# Gradient Descent

Gradient descent is an optimization algorithm used to minimize the loss function by adjusting the model parameters. Here's how it works in general terms:

Code Explanation:
1. Data: We use the inputs and targets arrays to train the model.
2. Model: The model performs a linear regression through the product of the input tensor by the transposed weights and the addition of the offset.
3. Error Function (MSE): We calculate the root mean square error (MSE) for predictions and target values.
4. Model Training: We train the model for 1000 epochs by updating weights and offsets with gradient descent.
5. Output: Every 100 epochs, we output the current error value (loss) to observe how it decreases.

Outcome:
1. After executing this code, it is possible to observe how the MSE decreases with each epoch, as well as how the weights change.
2. Too many epochs can lead to overfitting, while too few can lead to underfitting.

Conclusions:
1. Optimal Number of Epochs: For the data in the example, the optimal number of epochs can be around 500-1000, because after that, the accuracy of the model (MSE) stabilizes and decreases slightly.
2. MSE: With each epoch, MSE gradually decreases, showing that the model is becoming more accurate in predictions.
