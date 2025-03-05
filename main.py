import cv2
from video_llava import VideoLLaVA

# Load Video-LLaVA model
model = VideoLLaVA(model_path="path_to_model_weights")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get object description
    caption = model.describe(frame)
    print("Objects detected:", caption)
    
    # Show webcam feed
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
