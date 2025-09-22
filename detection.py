import os, json, cv2, numpy as np, tensorflow as tf
from ultralytics import YOLO
from collections import deque

MODEL_PATH = "resnet50_garbage_classifier_model.keras"
YOLO_WEIGHTS = "yolov8n.pt"
CLASS_INDICES_JSON = "class_indices.json"

CONF_THRESH = 0.6          # stricter threshold
SKIP_YOLO_CLASSES = {"person"}
MAX_BOX_AREA_FRAC = 0.35
MARGIN_FRAC = 0.12         # larger margin for more context
FRAME_SKIP = 2
INPUT_RESIZE = (1280, 720)

SMOOTH_WINDOW = 5

classifier = tf.keras.models.load_model(MODEL_PATH)
INPUT_SIZE = classifier.input_shape[1:3]

USE_RESNET_PREPROCESS = True
for layer in classifier.layers:
    if "rescaling" in layer.name.lower():
        USE_RESNET_PREPROCESS = False
        break
if USE_RESNET_PREPROCESS:
    from tensorflow.keras.applications.resnet50 import preprocess_input as RESNET_PREPROCESS

FALLBACK_CLASS_NAMES = ["cardboard","glass","metal","paper","plastic","trash"]
def load_class_names():
    if os.path.exists(CLASS_INDICES_JSON):
        with open(CLASS_INDICES_JSON,"r") as f:
            mapping = json.load(f)
        names = [None]*len(mapping)
        for name,idx in mapping.items(): names[idx]=name
        if all(isinstance(n,str) for n in names): return names
    return FALLBACK_CLASS_NAMES
CLASS_NAMES = load_class_names()

def safe_crop(img,x1,y1,x2,y2,margin_frac=0.0):
    H,W = img.shape[:2]; w,h = x2-x1,y2-y1
    dx,dy = int(w*margin_frac), int(h*margin_frac)
    x1=max(0,x1-dx); y1=max(0,y1-dy)
    x2=min(W-1,x2+dx); y2=min(H-1,y2+dy)
    return img[y1:y2,x1:x2]

def preprocess_for_classifier(crop):
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype("float32")
    if USE_RESNET_PREPROCESS: img = RESNET_PREPROCESS(img)
    else: img /= 255.0
    return np.expand_dims(img,axis=0)

def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam"); return
    detector = YOLO(YOLO_WEIGHTS)

    # Store predictions for smoothing
    pred_buffer = deque(maxlen=SMOOTH_WINDOW)

    print("Webcam started. Press 'q' to quit.")
    frame_count=0
    while True:
        ret,frame = cap.read()
        if not ret: break
        frame=cv2.resize(frame,INPUT_RESIZE)
        H,W=frame.shape[:2]; frame_area=H*W

        result=detector(frame,verbose=False)[0]
        names=result.names
        predictions=[]
        if frame_count%FRAME_SKIP==0:
            for b in result.boxes:
                x1,y1,x2,y2=map(int,b.xyxy[0])
                cls_id=int(b.cls[0]) if b.cls is not None else -1
                yolo_name=names.get(cls_id,"?")
                box_area=(x2-x1)*(y2-y1)
                if yolo_name in SKIP_YOLO_CLASSES: continue
                if box_area<=0 or box_area>MAX_BOX_AREA_FRAC*frame_area: continue
                obj=safe_crop(frame,x1,y1,x2,y2,margin_frac=MARGIN_FRAC)
                if obj.size==0: continue
                img=preprocess_for_classifier(obj)
                preds=classifier.predict(img,verbose=0)[0]
                predictions.append(((x1,y1,x2,y2),preds))
            if predictions: pred_buffer.append(predictions)

        # Use smoothed predictions (majority vote)
        if pred_buffer:
            # Take average of last K frames
            smoothed=[]
            for obj_idx in range(len(pred_buffer[-1])):
                # collect all probs for same object index
                all_preds=[buf[obj_idx][1] for buf in pred_buffer if len(buf)>obj_idx]
                mean_preds=np.mean(all_preds,axis=0)
                class_index=int(np.argmax(mean_preds))
                confidence=float(mean_preds[class_index])
                (x1,y1,x2,y2)=pred_buffer[-1][obj_idx][0]
                smoothed.append(((x1,y1,x2,y2),class_index,confidence))
            # Draw
            for (x1,y1,x2,y2),class_index,confidence in smoothed:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                if confidence>=CONF_THRESH:
                    label=f"{CLASS_NAMES[class_index]} ({confidence*100:.1f}%)"
                    print(f"Detected: {label}")
                else:
                    label="Unknown"
                cv2.putText(frame,label,(x1,max(y1-8,20)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        cv2.imshow("Stable Waste Detection + Classification",frame)
        frame_count+=1
        if cv2.waitKey(1)&0xFF==ord("q"): break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    run_webcam()
