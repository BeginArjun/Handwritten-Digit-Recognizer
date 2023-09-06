import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


np.set_printoptions(precision=2)

def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((20, 20))
    image.show()
    image_array = np.array(image)
    image_array=image_array/255.0
    return image_array

def predict_image(model, image_array):
    predictions = model.predict(image_array.reshape(1, 400))
    prediction_p = tf.nn.softmax(predictions)
    yhat = np.argmax(prediction_p)
    return yhat

def display_errors(model,X,y):
    f = model.predict(X)
    yhat = np.argmax(f, axis=1)
    doo = yhat != y[:,0]
    idxs = np.where(yhat != y[:,0])[0]
    if len(idxs) == 0:
        print("no errors found")
    else:
        cnt = min(8, len(idxs))
        fig, ax = plt.subplots(1,cnt, figsize=(5,1.2))
        fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.80]) #[left, bottom, right, top]

        for i in range(cnt):
            j = idxs[i]
            X_reshaped = X[j].reshape((20,20)).T

            # Display the image
            ax[i].imshow(X_reshaped, cmap='gray')

            # Predict using the Neural Network
            prediction = model.predict(X[j].reshape(1,400))
            prediction_p = tf.nn.softmax(prediction)
            yhat = np.argmax(prediction_p)

            # Display the label above the image
            ax[i].set_title(f"{y[j,0]},{yhat}",fontsize=10)
            ax[i].set_axis_off()
            fig.suptitle("Label, yhat", fontsize=12)
            fig.suptitle(f"{len(idxs)} errors out of {len(X)} images")
    return(len(idxs))


X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

tf.random.set_seed(1234)
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(units=25, activation='relu', name='L1'),
        tf.keras.layers.Dense(units=15, activation='relu', name='L2'), 
        tf.keras.layers.Dense(units=10, activation='linear', name='L3'),
    ], name="my_model"
)



model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)



model.fit(X_train, y_train, epochs=50, validation_split=0.1)

print( f"{display_errors(model,X,y)} errors out of {len(X)} images")

score=model.evaluate(X_train,y_train,verbose=0)
print("Average Cross-Validation Accuracy:", score)

test_predictions = np.argmax(model.predict(X_test), axis=1)

m, n = X_test.shape

fig, axes = plt.subplots(8,8, figsize=(5,5))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X_test[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Predict using the Neural Network
    prediction = model.predict(X_test[random_index].reshape(1,400))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)
    
    # Display the label above the image
    ax.set_title(yhat,fontsize=10)
    ax.set_axis_off()
fig.suptitle("yhat", fontsize=14)
plt.show()