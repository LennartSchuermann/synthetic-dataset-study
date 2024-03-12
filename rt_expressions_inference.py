import cv2
import torchvision.transforms as transforms
import onnxruntime as rt
import numpy as np

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
ort_session = rt.InferenceSession("trained_models/expr_net_s_synth_final.onnx")

classes = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

#cap = cv2.VideoCapture(0) # Used for realtime webcam capture
cap = cv2.VideoCapture("FaceTest.mp4") #https://www.youtube.com/watch?v=ZPB-Ov4zA4g

frame_index = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i in range(len(list(faces))):
        if i == 0:
            x, y, w, h = list(faces)[i]
            
            # Get Region Of Interest (Face)
            ROI = gray_frame[y:y+h, x:x+w]
            ROI_SAFE = ROI
            ROI = cv2.resize(ROI, (48, 48))
            
            to_tensor = transforms.ToTensor()
            ROI = to_tensor(ROI)
            ROI.unsqueeze_(0)
            
            # Inference
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(ROI)}
            ort_outs = ort_session.run(None, ort_inputs)
            
            prediction = classes[np.argmax(ort_outs)]
            #cv2.imwrite("application_eval/synth/" + str(frame_index) + "_" + prediction + ".png", ROI_SAFE) # Save ROI

            # UI
            cv2.putText(frame, text=prediction, org=(x, y + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(0, 0, 255), thickness=2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            frame_index += 1
        

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection - L. Schürmann', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()