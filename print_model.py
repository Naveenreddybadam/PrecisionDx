from tensorflow.keras.models import load_model
m = load_model('tumor_type_classifier.h5')
print('model inputs type:', type(m.inputs), 'len:', len(m.inputs))
print('input names:', [getattr(x,'name',None) for x in m.inputs])
print('Top-level layer count:', len(m.layers))
for i,layer in enumerate(m.layers):
    print(i, type(layer), layer.name)
