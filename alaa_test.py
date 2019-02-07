import tensorflow as tf
from train import Train
import numpy as np
import cv2
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
#%matplotlib inline
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

#get webcam frames 
print("starting wbcam")
cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)
#resized_img = cv2.resize(np_img, (320, 240))

#test_img = scipy.misc.imread('./test_data/test_0.jpg', mode="RGB")
test_img = cv2.imread('./dataset/5.jpg')
plt.imshow(test_img)
plt.show()


model = Train()
model.build_graph()
model_in = model.input
model_out_box = model.out_box
model_out_has_obj = model.out_has_obj
# Load tensorflow section
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "./save/model")
print('Number of parameters:',model.num_parameters)


box, has_obj = sess.run([model_out_box, model_out_has_obj],feed_dict={model_in: [test_img]})
print(box)
print(has_obj)

y,x,w,h = box[0]
x = int(x*320.0)
y = int(y*240.0)
w = int(w*320)
h = int(h*240)
print("x: %d, y: %d, w: %d h: %d" % (x,y,w,h))
cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
plt.imshow(test_img)
plt.show()


while(True):
    ret, img = cap.read()
    
    
    #image = np.array(img).reshape(1, 512,1800,3)  #(1, 240,320,3)
    box, has_obj = sess.run([model_out_box, model_out_has_obj],feed_dict={model_in: [img]})

    y,x,w,h = box[0]
    x = int(x*320.0)
    y = int(y*240.0)
    w = int(w*320)
    h = int(h*240)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Video', img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break