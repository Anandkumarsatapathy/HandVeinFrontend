import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('handVein.h5')

# Define the class labels
classes = ['Authorized User', 'UnAuthorized User']
# Define the function to make a prediction on the input image


def predict(image):
    # Load and preprocess the input image
    img = Image.open(image).resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)

    # Make a prediction
    pred = model.predict(img)
    pred_class = classes[pred.argmax()]
    pred_prob = pred.max()

    return pred_class, pred_prob

# Define the Streamlit app


def app():
    st.title('HandVein Image Detection')
    st.write('Upload a HandVein Image')

    # Upload the input image
    uploaded_file = st.file_uploader(
        'Choose an image...', type=['jpg', 'jpeg', 'png'])

    # Make a prediction if an image is uploaded
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        pred_class, pred_prob = predict(uploaded_file)
        st.title(f'{pred_class}')


# Run the Streamlit app
if __name__ == '__main__':
    app()
