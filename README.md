## Project Steps

1. Read the IMDb review dataset from a CSV file.
2. Clean the review text by removing HTML tags, punctuation, and special characters.
3. Convert sentiment labels into numerical values: positive = 1 and negative = 0.
4. Split the dataset into training and testing sets.
5. Tokenize the movie reviews into numerical sequences.
6. Pad all reviews to the same length.
7. Build an RNN model using an embedding layer, SimpleRNN layer, dropout layer, and dense output layer.
8. Train the model using binary crossentropy and Adam optimizer.
9. Evaluate the model on the test data.
10. Test the trained model with a new sample movie review.
