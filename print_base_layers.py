from tensorflow.keras.models import load_model
m = load_model('tumor_type_classifier.h5')
base = m.get_layer('mobilenetv2_1.00_224')
print('base layer count:', len(base.layers))
for i,layer in enumerate(base.layers[-40:]):
    print(i-40, type(layer), layer.name)
