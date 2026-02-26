import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
m = load_model('tumor_type_classifier.h5')
img = image.load_img('test.jpg', target_size=(224,224))
arr = image.img_to_array(img)
arr = np.expand_dims(arr,0)/255.0
print('model expects inputs:', type(m.inputs), [x.shape for x in m.inputs])
print('calling predict...')
print(m.predict(arr).shape)
