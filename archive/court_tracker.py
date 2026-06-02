import cv2
import numpy as np
import json

class CourtTracker:
    def __init__(self, video_path, output_json="court_points.json"):
        self.cap = cv2.VideoCapture(video_path)
        self.output_json = output_json
        self.width, self.height = 960, 540
        self.points = []
        self.net_roi = None
        
        # Initialize SIFT detector (robust to light/shadow)
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
        
        self.ref_kp = None
        self.ref_des = None
        self.ref_pts = None # Initial user points
        self.data_log = []

    def select_initial_data(self, frame):
        """UI for selecting the 4 points and the Net ROI."""
        print("1. Click the 4 baseline/alley intersections.")
        print("2. Press 'Enter' when done.")
        
        temp_pts = []
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                temp_pts.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Initialization", frame)

        cv2.namedWindow("Initialization")
        cv2.setMouseCallback("Initialization", mouse_callback)
        cv2.imshow("Initialization", frame)
        cv2.waitKey(0)
        
        self.ref_pts = np.array(temp_pts, dtype=np.float32).reshape(-1, 1, 2)
        
        print("3. Drag a box around the Net/Net Posts (Anchor Zone).")
        self.net_roi = cv2.selectROI("Initialization", frame, fromCenter=False)
        cv2.destroyAllWindows()

    def get_features(self, frame):
        """Extract features specifically from the Net ROI."""
        x, y, w, h = [int(v) for v in self.net_roi]
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        kp, des = self.sift.detectAndCompute(frame, mask)
        return kp, des

    def run(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv2.resize(frame, (self.width, self.height))
        self.select_initial_data(frame.copy())
        
        # Set reference features from Frame 0
        self.ref_kp, self.ref_des = self.get_features(frame)
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (self.width, self.height))
            frame_idx += 1
            
            curr_kp, curr_des = self.get_features(frame)
            
            if curr_des is not None and len(curr_des) > 10:
                # Match features between reference and current
                matches = self.matcher.knnMatch(self.ref_des, curr_des, k=2)
                
                # Lowe's ratio test
                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                
                if len(good_matches) > 15:
                    src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Find Homography (USAC_MAGSAC is best for outliers/nudge detection)
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
                    
                    if H is not None:
                        # Transform the 4 intersection points
                        new_pts = cv2.perspectiveTransform(self.ref_pts, H)
                        
                        # Update reference features to handle slow drift
                        # (Optional: Only update every X frames)
                        self.ref_kp, self.ref_des = curr_kp, curr_des
                        self.ref_pts = new_pts
                        
                        # Log results
                        self.data_log.append({
                            "frame": frame_idx,
                            "points": new_pts.reshape(-1, 2).tolist()
                        })
                        
                        # Visual Debug
                        for p in new_pts:
                            cv2.circle(frame, (int(p[0][0]), int(p[0][1])), 5, (0, 0, 255), -1)
                    else:
                        self.trigger_reinit(frame)
                else:
                    self.trigger_reinit(frame)
            else:
                self.trigger_reinit(frame)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        with open(self.output_json, 'w') as f:
            json.dump(self.data_log, f)
            
        self.cap.release()
        cv2.destroyAllWindows()

    def trigger_reinit(self, frame):
        """Prompt user for new points if tracking is lost."""
        print(f"Tracking lost on frame. Please re-initialize.")
        self.select_initial_data(frame)
        self.ref_kp, self.ref_des = self.get_features(frame)

if __name__ == "__main__":
    tracker = CourtTracker("/Volumes/Anya/Data/43/snippet.mp4")
    tracker.run()