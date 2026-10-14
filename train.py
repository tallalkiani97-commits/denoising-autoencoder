"""Small denoising autoencoder training starter."""
import tensorflow as tf
from tensorflow.keras import layers, Model

(x_train,_),(x_test,_)=tf.keras.datasets.cifar10.load_data()
x_train=x_train.astype("float32")/255.; x_test=x_test.astype("float32")/255.
noise=.18*tf.random.normal(tf.shape(x_train)); noisy=tf.clip_by_value(x_train+noise,0,1)
inputs=layers.Input((32,32,3)); x=layers.Conv2D(16,3,activation="relu",padding="same")(inputs); x=layers.MaxPooling2D()(x); x=layers.Conv2D(8,3,activation="relu",padding="same")(x); x=layers.UpSampling2D()(x); outputs=layers.Conv2D(3,3,activation="sigmoid",padding="same")(x)
model=Model(inputs,outputs); model.compile(optimizer="adam",loss="mse")
model.fit(noisy,x_train,epochs=1,batch_size=128,validation_split=.05); print("trained",model.evaluate(x_test,x_test,verbose=0))
