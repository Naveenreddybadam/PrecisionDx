import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
m = load_model('tumor_type_classifier.h5')
arr = image.img_to_array(image.load_img('test.jpg', target_size=(224,224)))
arr = np.expand_dims(arr,0)/255.0
base = m.get_layer('mobilenetv2_1.00_224')
conv_tensor = base.get_layer('Conv_1').output
from tensorflow.keras.models import Model
grad_model = Model(inputs=m.inputs, outputs=[conv_tensor, m.outputs[0]])
print('grad_model built, calling...')
outs = grad_model.predict(arr)
print(type(outs), len(outs))
print([o.shape for o in outs])
