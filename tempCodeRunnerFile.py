def predpneumoniaPrediction(image):
    # Preprocess the image
    image= preprocess_image(image)
    # Make a prediction using the model
    model_pred = model.predict(image)
    probability = model_pred[0]
    if probability[0] > 0.5:
        result = 'Pneumonia POSITIVE. Consult Radiologist as soon as possible.'
    else:
        result = 'Pneumonia NEGATIVE. You have a Healthy lung.'
        
   
    # Return the predicted class and processed images
    return result