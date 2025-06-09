import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or your own trained model

from boxmot.trackers.botsort.botsort import BotSort

from pathlib import Path
import torch

# Assuming BotSort class is imported from your tracking package

# Device setup: use GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to the ReID weights file (make sure you have this)
reid_weights_path = Path('osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth')


# Open webcam or video file
cap = cv2.VideoCapture("test_vid.mp4")  # Replace with 'video.mp4' to use a video file

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0


# Create VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))

tracker = BotSort(
    reid_weights=reid_weights_path,  # Path to your appearance model weights
    device=device,                   # Run on GPU or CPU
    half=True,                      # Use FP16 for faster ReID inference (recommended if device is GPU)
    per_class=False,                # Track all classes together (True if you want separate tracking per class)
    
    track_high_thresh=0.6,          # Confidence threshold for high-quality detections in first association
    track_low_thresh=0.1,           # Lower confidence threshold for second association to recover missed tracks
    new_track_thresh=0.7,           # Minimum confidence required to initialize a new track
    
    track_buffer=30,                # Keep lost tracks for 30 frames before removal (helps smooth occlusions)
    match_thresh=0.8,               # Matching threshold to associate detections with existing tracks
    
    proximity_thresh=0.5,           # IoU threshold to filter matches early (reject unlikely matches)
    appearance_thresh=0.5,         # Max appearance distance for matching tracks and detections (ReID similarity)
    
    cmc_method='orb',    # Global Motion Compensation method to handle camera movement
    frame_rate=fps,                 # Video FPS, used to scale buffer and motion compensation
    
    fuse_first_associate=True,     # Fuse motion and appearance scores for stronger first association
    
    with_reid=True                 # Use appearance features for re-identification
)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)[0]  # Get results from first model output

    # Extract detections in the required format for the tracker
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        dets.append([x1, y1, x2, y2, conf, cls])

    # Convert to torch.Tensor if not empty
    dets_tensor = torch.tensor(dets) if dets else torch.zeros((0, 6))

    # Run BotSort tracker
    tracks = tracker.update(dets_tensor.cpu().numpy(), frame)


    # Draw results
    for track in tracks:
        #print(track)
        x1, y1, x2, y2, track_id, cls, conf_score = track[:7]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = f'ID {int(track_id)}'

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save frame to video
    out.write(frame)

    # Show the frame (optional)
    cv2.imshow("Tracked", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
