from packeges import *

class StereoYOLOProcessor:
    def __init__(self, model_path="last.pt", config_file="config.txt", confidence_threshold=None):
        self.model = YOLO(model_path)
        
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            print(f"Loaded overridden confidence threshold: {self.confidence_threshold} (from __init__ parameter)")
        else:
            self.confidence_threshold = self._load_config(config_file) 
            print(f"Loaded confidence threshold: {self.confidence_threshold} from {config_file}")

        self._init_stereo_matcher() 
        
    def _init_stereo_matcher(self):
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64, # חייב להיות כפולה של 16
            blockSize=5,     
            P1=8*3*5**2,     
            P2=32*3*5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY 
        )
    
    def _load_config(self, config_file_path): 
        default_confidence = 0.8
        try:
            with open(config_file_path, "r") as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("confidence="):
                        value_str = line.split("=")[1]
                        try:
                            confidence = float(value_str)
                            print(f"Loaded confidence threshold: {confidence} from {config_file_path}")
                            return confidence
                        except ValueError:
                            print(f"Warning: Invalid confidence value in {config_file_path}. Using default: {default_confidence}")
                            return default_confidence
                print(f"Warning: 'confidence' not found in {config_file_path}. Using default: {default_confidence}")
                return default_confidence
        except FileNotFoundError:
            print(f"Warning: Configuration file '{config_file_path}' not found. Using default confidence: {default_confidence}")
            return default_confidence
        except Exception as e:
            print(f"Error loading confidence threshold from {config_file_path}: {e}. Using default: {default_confidence}")
            return default_confidence
        
    def extract_calibration_data(self, calib_file_path):
        """Extract baseline and focal length from KITTI calibration file"""
        baseline = None
        focal_length = None
        
        try:
            with open(calib_file_path, "r") as file:
                for line in file:
                    if "P_rect_02:" in line or "P2:" in line:
                        values = line.split()
                        if len(values) >= 4:
                            focal_length = float(values[1])
                            if len(values) >= 4 and float(values[4]) != 0:
                                baseline = abs(float(values[4]) / focal_length)
                    elif "baseline:" in line.lower():
                        values = line.split()
                        if len(values) > 1:
                            baseline = float(values[1])
                            
            if baseline is None:
                baseline = 0.062 
                print(f" Using default KITTI baseline: {baseline}m")
                
            if focal_length is None:
                raise ValueError("Could not extract focal length from calibration file")
                
        except Exception as e:
            print(f"Error reading calibration: {e}")
            focal_length = 721.5 
            baseline = 0.062 
            print(f" Using default KITTI parameters: f={focal_length}, b={baseline}")
            
        return baseline, focal_length

    def detect_stop_sign(self, img):
        """
        מבצע זיהוי תמרורי עצור בתמונה באמצעות מודל YOLO.
        מחזיר מילון עם נתוני התיבה התוחמת והביטחון של תמרור העצור הראשון שזוהה
        שעובר את סף הביטחון, או None אם לא נמצא.
        """
        results = self.model(img, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                
                if name.lower() in ['stop', 'stop sign']:
                    confidence = float(box.conf[0]) 
                    print(f"   [DEBUG] Stop sign candidate: Conf={confidence:.3f}, Class={name}")
                    
                    if confidence > self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        return {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        }
        return None

    def compute_disparity(self, img_left, img_right):
        """Compute disparity map with preprocessing"""
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        gray_left = cv2.equalizeHist(gray_left)
        gray_right = cv2.equalizeHist(gray_right)
        
        disparity = self.stereo_matcher.compute(gray_left, gray_right)
        
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity

    def calculate_depth(self, disparity_map, detection, baseline, focal_length):
        if detection is None or disparity_map is None:
            return None

        x1, y1, x2, y2 = detection['bbox']

        img_height, img_width = disparity_map.shape[:2]
        pad_x = 5 
        pad_y = 5 
        
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(img_width, x2 + pad_x)
        y2_padded = min(img_height, y2 + pad_y)

        disparity_region = disparity_map[y1_padded:y2_padded, x1_padded:x2_padded] 

        valid_disparities = disparity_region[disparity_region > 0]

        if len(valid_disparities) == 0:
            return None

        disparity_value = np.median(valid_disparities)

        if disparity_value > 0:
            depth = (baseline * focal_length) / disparity_value
            return depth

        return None

    def visualize_results(self, img_left, img_right, disparity, detection_left, detection_right, depth, wait_for_key=True, display_scale=0.5):
        """Create visualization using OpenCV (no matplotlib dependency)"""
        
        # Scaling images for display
        if img_left is not None:
            h, w = img_left.shape[:2]
            img_left_resized = cv2.resize(img_left, (int(w * display_scale), int(h * display_scale)))
        else:
            img_left_resized = np.zeros((int(480 * display_scale), int(640 * display_scale), 3), dtype=np.uint8)

        if img_right is not None:
            h, w = img_right.shape[:2]
            img_right_resized = cv2.resize(img_right, (int(w * display_scale), int(h * display_scale)))
        else:
            img_right_resized = np.zeros((int(480 * display_scale), int(640 * display_scale), 3), dtype=np.uint8)

        display_left = img_left_resized.copy()
        display_right = img_right_resized.copy()
        
        # Draw detection on left image
        if detection_left:
            x1, y1, x2, y2 = [int(coord * display_scale) for coord in detection_left['bbox']]
            center_x, center_y = [int(coord * display_scale) for coord in detection_left['center']]
            
            cv2.rectangle(display_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display_left, (center_x, center_y), int(5 * display_scale), (0, 0, 255), -1) 
            
            depth_text = f"Depth: {depth:.2f}m" if depth is not None else "No depth"
            cv2.putText(display_left, depth_text, (x1, y1 - int(10 * display_scale)),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7 * display_scale, (0, 255, 0), int(2 * display_scale))

        # Draw detection on right image (if any)
        if detection_right:
            x1_r, y1_r, x2_r, y2_r = [int(coord * display_scale) for coord in detection_right['bbox']]
            cv2.rectangle(display_right, (x1_r, y1_r), (x2_r, y2_r), (0, 255, 0), 2)
            
        # Prepare disparity map for display
        disp_to_display = None
        if disparity is not None and disparity.size > 0:
            print(f"   [DEBUG] Disparity map max value: {np.max(disparity):.2f}")
            if np.max(disparity) > 0:
                disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                disp_to_display = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
                disp_to_display = cv2.resize(disp_to_display, (display_left.shape[1], display_left.shape[0]))
                # **** שורה זו הוסרה כדי לבטל את היפוך הדיספרטיות ****
                # cv2.flip(disp_to_display, 1) 
            else:
                # If disparity map is all zeros
                disp_to_display = np.zeros_like(display_left, dtype=np.uint8) # Create a black image
                cv2.putText(disp_to_display, "Zero Disparity Map (No Depth)", (int(50 * display_scale), disp_to_display.shape[0] // 2),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7 * display_scale, (255, 255, 255), int(2 * display_scale))
        else:
            # If no disparity data is available
            disp_to_display = np.zeros_like(display_left, dtype=np.uint8) # Create a black image
            cv2.putText(disp_to_display, "No Disparity Data Available", (int(50 * display_scale), disp_to_display.shape[0] // 2),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7 * display_scale, (255, 255, 255), int(2 * display_scale))
        
        window_name_left = "Left Image + Detection"
        window_name_right = "Right Image + Detection"
        window_name_disparity = "Disparity Map"

        # Create windows and display images
        cv2.namedWindow(window_name_left, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_name_right, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_name_disparity, cv2.WINDOW_NORMAL)

        cv2.imshow(window_name_left, display_left)
        cv2.imshow(window_name_right, display_right)
        cv2.imshow(window_name_disparity, disp_to_display)

        # Set window positions and size explicitly for better control
        # We can calculate a default window size based on the scaled image dimensions
        default_window_w = display_left.shape[1]
        default_window_h = display_left.shape[0]

        cv2.resizeWindow(window_name_left, default_window_w, default_window_h)
        cv2.resizeWindow(window_name_right, default_window_w, default_window_h)
        cv2.resizeWindow(window_name_disparity, default_window_w, default_window_h)

        # Positioning windows side by side, or in a stack if side-by-side causes issues
        # For a stacked view, assuming the display_scale gives reasonable sizes:
        cv2.moveWindow(window_name_left, 10, 10) # Top-left
        cv2.moveWindow(window_name_right, 10, 10 + default_window_h + 30) # Below left image
        cv2.moveWindow(window_name_disparity, 10, 10 + (default_window_h + 30) * 2) # Below right image


        if wait_for_key:
            print(f" לחץ על מקש כלשהו כדי להמשיך...")
            cv2.waitKey(2000) 
        else:
            cv2.waitKey(2000) 
        cv2.destroyAllWindows()

    def process_scene(self, scene_dir):
        """Process a single scene directory"""
        print(f"\n Processing {scene_dir}...")

        calib_files = glob.glob(os.path.join(scene_dir, "*.txt"))
        if not calib_files:
            print(f" No calibration file found in {scene_dir}")
            return 

        calib_file_path = calib_files[0]
        scene_key = os.path.splitext(os.path.basename(calib_file_path))[0]

        try:
            baseline, focal_length = self.extract_calibration_data(calib_file_path)
            print(f" Calibration: Baseline={baseline:.3f}m, Focal Length={focal_length:.1f}px")
        except Exception as e:
            print(f" Calibration error: {e}")
            return 

        left_folder = os.path.join(scene_dir, "left")
        right_folder = os.path.join(scene_dir, "right")

        if not os.path.exists(left_folder) or not os.path.exists(right_folder):
            print(f" Missing left or right folder in {scene_dir}")
            return 

        left_images = sorted(glob.glob(os.path.join(left_folder, f"{scene_key}*.png")))
        right_images = sorted(glob.glob(os.path.join(right_folder, f"{scene_key}*.png")))

        if len(left_images) != len(right_images):
            print(f" Image count mismatch: {len(left_images)} left, {len(right_images)} right")
            return 

        for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
            print(f"    Processing image pair {idx+1}/{len(left_images)}")

            img_left = cv2.imread(left_path)
            img_right = cv2.imread(right_path)

            detection_left = None
            detection_right = None
            disparity_map = None 
            depth = None 
            status_message = ""
            
            should_wait_for_key = True 

            if img_left is None or img_right is None:
                status_message = "Failed to load images."
                dummy_shape = (int(480 * 0.5), int(640 * 0.5)) 
                disparity_map = np.zeros(dummy_shape, dtype=np.float32)
                should_wait_for_key = False 

            else:
                detection_left = self.detect_stop_sign(img_left)
                detection_right = self.detect_stop_sign(img_right)

                if detection_left is None:
                    status_message = "No stop sign detected in left image."
                    should_wait_for_key = False
                elif detection_right is None:
                    status_message = "Stop sign detected in left, but not in right. Cannot calculate depth."
                    should_wait_for_key = False
                else:
                    left_center_x, left_center_y = detection_left['center']
                    right_center_x, right_center_y = detection_right['center']
                    
                    if abs(left_center_y - right_center_y) > 10 or (right_center_x >= left_center_x and abs(left_center_y - right_center_y) <= 10):
                        status_message = (f"Stop sign detected in both, but positions are inconsistent. "
                                          f"Cannot calculate depth (disparity X: {left_center_x - right_center_x:.2f}, Y diff: {abs(left_center_y - right_center_y):.2f}).")
                        should_wait_for_key = False
                    else:
                        disparity_map = self.compute_disparity(img_left, img_right)
                        depth = self.calculate_depth(disparity_map, detection_left, baseline, focal_length)
                        
                        if depth is None:
                            status_message = (f"Stop sign detected (confidence left: {detection_left['confidence']:.3f}, right: {detection_right['confidence']:.3f}). "
                                              f"Could not calculate valid depth.")
                            should_wait_for_key = False
                        else:
                            status_message = (f"Stop sign detected! Depth: {depth:.2f} meters "
                                              f"(confidence left: {detection_left['confidence']:.3f}, right: {detection_right['confidence']:.3f}).")
                            should_wait_for_key = True
                
            print(f"    {status_message}")
            
            if disparity_map is None and img_left is not None:
                dummy_shape = (int(img_left.shape[0] * 0.5), int(img_left.shape[1] * 0.5))
                disparity_map = np.zeros(dummy_shape, dtype=np.float32)

            if img_left is not None and img_right is not None:
                self.visualize_results(img_left, img_right, disparity_map, 
                                        detection_left, detection_right, depth, wait_for_key=should_wait_for_key)
            else:
                print("    Skipping visualization due to image loading failure.")
            
def main():
    processor = StereoYOLOProcessor(model_path="last.pt", config_file="config.txt") 
    
    scene_dirs = ["scene0", "scene1", "scene68", "scene141"] 
    
    for scene_dir in scene_dirs:
        if os.path.exists(scene_dir):
            processor.process_scene(scene_dir)
        else:
            print(f" Scene directory {scene_dir} not found")

if __name__ == "__main__":
    main()