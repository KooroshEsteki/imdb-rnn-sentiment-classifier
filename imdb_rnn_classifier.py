import re
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


# 
# Step 1: Read the IMDb dataset
#

dataset_path = "IMDB Dataset.csv"

df = pd.read_csv(dataset_path)

print("Dataset shape:", df.shape)
print(df.head())


# 
# Step 2: Clean the text reviews
#

def clean_review(text):
    """
    This function cleans each movie review by:
    - removing HTML tags
    - removing special characters
    - converting text to lowercase
    """

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["review"] = df["review"].apply(clean_review)


# 
# Step 3: Convert sentiment labels to numbers
# positive = 1, negative = 0
# 

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})


# 
# Step 4: Split data into training and testing sets
# 

x = df["review"]
y = df["sentiment"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))


#
# Step 5: Tokenize the reviews
# 

vocab_size = 10000
max_review_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)

x_train_sequences = tokenizer.texts_to_sequences(x_train)
x_test_sequences = tokenizer.texts_to_sequences(x_test)


# 
# Step 6: Pad the sequences
# 

x_train_padded = pad_sequences(
    x_train_sequences,
    maxlen=max_review_length,
    padding="post",
    truncating="post"
)

x_test_padded = pad_sequences(
    x_test_sequences,
    maxlen=max_review_length,
    padding="post",
    truncating="post"
)


#
# Step 7: Build the RNN model
# 

embedding_dimension = 100

model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dimension,
        input_length=max_review_length
    )
)

model.add(SimpleRNN(100))
model.add(Dropout(0.2))

model.add(
    Dense(
        units=1,
        activation="sigmoid"
    )
)


#
# Step 8: Compile the model
# 

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# 
# Step 9: Train the model
#

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    x_train_padded,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)


# 
# Step 10: Test the model
# 

test_loss, test_accuracy = model.evaluate(x_test_padded, y_test)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


#
# Step 11: Test with a new review
# 

new_review = [
    "This movie was amazing. The story was great and the acting was excellent."
]

new_review_cleaned = [clean_review(review) for review in new_review]
new_review_sequence = tokenizer.texts_to_sequences(new_review_cleaned)

new_review_padded = pad_sequences(
    new_review_sequence,
    maxlen=max_review_length,
    padding="post",
    truncating="post"
)

prediction = model.predict(new_review_padded)

print("\nPrediction value:", prediction[0][0])

if prediction[0][0] >= 0.5:
    print("Predicted Sentiment: Positive")
else:
    print("Predicted Sentiment: Negative")


# The trained model object
model
