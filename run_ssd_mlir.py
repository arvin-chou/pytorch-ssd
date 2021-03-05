import numpy as np
import cv2
import sys
import torch

import pymlir
import pyruntime

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <mlir|cvimodel> <model path> <label path> <image path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]

class_names = [name.strip() for name in open(label_path).readlines()]
if net_type == "mlir":
    net = pymlir.module()
    print('mlir load module ', model_path)
    net.load(model_path)
    print('load module done')
elif net_type == "cvimodel":
    batch_size = 1
    print('cvimodel load module ', model_path)
    net = pyruntime.Model(model_path, batch_size, output_all_tensors=False)
else:
    print("The net type is wrong. It should be mlir|cvimodel")
    sys.exit(1)

orig_image = cv2.imread(image_path)
orgimage = orig_image
image = preprocess(orgimage)
image = image.astype(np.float32)
image = image.transpose((2, 0, 1))

def postprocess(img, out, prob_threshold=0.01):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]

    return (box.astype(np.int32), conf, cls)


if net_type == "mlir":
    res = net.run(image)
    out = net.get_all_tensor()
    box, conf, cls = postprocess(orgimage, out)
    labels = torch.from_numpy(cls)
    probs = torch.from_numpy(conf)
    boxes = torch.from_numpy(box)
elif net_type == "cvimodel":
    # fill data to inputs
    data = net.inputs[0].data
    qscale = net.inputs[0].qscale
    # load input data and quant to int8
    input = quant(image, qscale)
    # fill input data to input tensor of model
    data[:] = input.reshape(data.shape)
    # forward
    net.forward()

    out = {"detection_out": net.outputs[0].data}
    box, conf, cls = postprocess(orgimage, out)
    labels = torch.from_numpy(cls)
    probs = torch.from_numpy(conf)
    boxes = torch.from_numpy(box)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{class_names[labels[i].int()]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")
