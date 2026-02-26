import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D


def _find_last_conv_layer(model):
    # Search direct children first for Conv2D
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return (model, layer.name)

    # Search nested models (e.g., MobileNetV2 base) and return parent model + layer name
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for inner in reversed(layer.layers):
                if isinstance(inner, Conv2D):
                    return (layer, inner.name)
                if "conv" in inner.name.lower() and hasattr(inner, "kernel"):
                    return (layer, inner.name)

    # Fallback: search all layers by name containing 'conv'
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower() and hasattr(layer, "kernel"):
            return (model, layer.name)

    return (None, None)


def make_gradcam(img_path="test.jpg", model_path="tumor_type_classifier.h5",
                 output_path="gradcam_output.jpg", target_size=(224, 224)):
    model = load_model(model_path)

    # Ensure model is built
    dummy = tf.zeros((1, target_size[0], target_size[1], 3))
    _ = model(dummy)

    parent_model, last_conv_name = _find_last_conv_layer(model)
    if parent_model is None or last_conv_name is None:
        raise ValueError("Could not find a Conv2D layer in the model.")

    # Prefer the typical MobileNetV2 base + 'Conv_1' pattern if present
    last_conv_tensor = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower() or 'mobilenetv2' in layer.name.lower():
            try:
                last_conv_tensor = layer.get_layer('Conv_1').output
                break
            except Exception:
                pass

    if last_conv_tensor is None:
        # Resolve the target layer from the main model if possible (preferred),
        # otherwise fall back to the nested parent model's layer.
        try:
            target_layer = model.get_layer(last_conv_name)
        except (ValueError, Exception):
            target_layer = parent_model.get_layer(last_conv_name)

        # Get the tensor for the conv layer that is connected into the main model graph
        last_conv_tensor = target_layer.output

    # Load and preprocess image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    input_array = np.expand_dims(img_array, axis=0) / 255.0

    # Try Grad-CAM; if anything fails (nested-functional graph issues),
    # fall back to a gradient-wrt-input saliency map.
    try:
        # Compute gradients by performing a manual forward pass so the conv outputs
        # and predictions are computed in the same graph/tape (handles nested models).
        input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            x = input_tensor
            conv_outputs = None
            # Walk through top-level layers; if a nested model is encountered,
            # walk through its inner layers to capture the target conv output.
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    for inner in layer.layers:
                        x = inner(x)
                        if inner.name == last_conv_name:
                            conv_outputs = x
                else:
                    x = layer(x)

            predictions = x
            if conv_outputs is None:
                raise ValueError('Could not capture conv layer output during forward pass')

            tape.watch(conv_outputs)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        # Weighted sum of feature maps
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val is None or max_val == 0:
            heatmap = heatmap.numpy()
        else:
            heatmap = heatmap / (max_val + 1e-10)
            heatmap = heatmap.numpy()

    except Exception as e:
        print('Grad-CAM failed, falling back to input saliency:', e)
        # Saliency: gradients w.r.t. input
        input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
        print('Using saliency fallback')
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            # model may expect a list input structure
            if isinstance(model.inputs, (list, tuple)):
                preds = model([input_tensor])
            else:
                preds = model(input_tensor)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            print('preds shape (eager):', getattr(preds, 'shape', None))
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, input_tensor)[0]  # shape (H, W, C)
        saliency = tf.reduce_mean(tf.abs(grads), axis=-1).numpy()
        saliency = np.maximum(saliency, 0)
        saliency = saliency / (np.max(saliency) + 1e-10)
        heatmap = saliency

    # Load original image (BGR)
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, target_size)

    heatmap = cv2.resize(heatmap, target_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    written = cv2.imwrite(output_path, superimposed)
    print('Wrote output:', output_path, 'success=', written)

    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Grad-CAM for an image")
    parser.add_argument("--img", default="test.jpg")
    parser.add_argument("--model", default="tumor_type_classifier.h5")
    parser.add_argument("--out", default="gradcam_output.jpg")
    args = parser.parse_args()

    make_gradcam(img_path=args.img, model_path=args.model, output_path=args.out)