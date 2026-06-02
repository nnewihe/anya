import cv2
import os
import pandas as pd

# Path configuration
BASE_PATH = "/Volumes/Anya/Data/"
MATCH_IDS = [f"{i:02d}" for i in range(1, 50)]  # Adjust range as needed
CROP_FPS = 30  # Assuming 30fps, 2 seconds = 60 frames

labels = []

def label_video(match_id):
    video_path = os.path.join(BASE_PATH, match_id, "snippet.mp4")
    if not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"--- Processing Match {match_id} ---")
    print("Controls: [W] Walk Onset | [N] No Walk | [Q] Next Video | [Space] Pause")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame / fps

        # Display instructions on frame
        cv2.putText(frame, f"Match: {match_id} | Time: {current_time:.2f}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Labeler', frame)

        key = cv2.waitKey(25) & 0xFF
        
        if key == ord('w'):
            print(f"Labeled WALKING at {current_time:.2f}s")
            labels.append({'match': match_id, 'timestamp': current_time, 'label': 1})
        
        elif key == ord('n'):
            print(f"Labeled NO-WALK at {current_time:.2f}s")
            labels.append({'match': match_id, 'timestamp': current_time, 'label': 0})
            
        elif key == ord(' '):  # Pause
            cv2.waitKey(-1)
            
        elif key == ord('q'):
            break

    cap.release()

# Run for all matches
for m_id in MATCH_IDS:
    label_video(m_id)

cv2.destroyAllWindows()

# Save labels to CSV
df = pd.DataFrame(labels)
df.to_csv("walking_labels.csv", index=False)
print("Labels saved to walking_labels.csv")