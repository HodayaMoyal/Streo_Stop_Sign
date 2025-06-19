from packeges import *
import glob 

class StereoYOLOProcessor:
    def __init__(self, model_path="last.pt", confidence_threshold=0.8):
        # Hardcoded values as per "without config file" request
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        
        self.confidence_threshold = confidence_threshold
        
        self._init_stereo_matcher() 
        
    def _init_stereo_matcher(self):
        """Initializes the StereoSGBM matcher with hardcoded parameters."""
        # Hardcoded stereo parameters
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,        
            numDisparities=160,      
            blockSize=9,              
            P1=8 * 3 * 9**2,          
            P2=32 * 3 * 9**2,         
            disp12MaxDiff=1,      
            uniquenessRatio=10,    
            speckleWindowSize=100,
            speckleRange=32,      
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
    def extract_calibration_data(self, calib_file_path):
        """Extract baseline and focal length from KITTI calibration file or use hardcoded defaults."""
        # Hardcoded default calibration values
        default_baseline = 0.12  # meters, a common range for car stereo cameras
        default_focal_length = 800.0 # pixels

        baseline = None
        focal_length = None
        
        try:
            if calib_file_path and os.path.exists(calib_file_path):
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
                baseline = default_baseline
                print(f" Using default estimated baseline: {baseline}m")
                
            if focal_length is None:
                focal_length = default_focal_length
                print(f" Using default estimated focal length: {focal_length}px")
                
        except Exception as e:
            print(f"Error reading calibration: {e}")
            focal_length = default_focal_length
            baseline = default_baseline
            print(f" Using default estimated parameters: f={focal_length}, b={baseline}")
            
        return baseline, focal_length

    def detect_stop_sign(self, img):
        """
        Performs stop sign detection in an image using the YOLO model.
        Returns a dictionary with the bounding box and confidence data of the first detected stop sign
        that passes the confidence threshold, or None if no stop sign is found.
        """
        results = self.model(img, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                
                if name.lower() in ['stop', 'stop sign']: 
                    confidence = float(box.conf[0]) 
                    print(f"       [DEBUG] Stop sign candidate: Conf={confidence:.3f}, Class={name}")
                    
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
        
        gray_left = cv2.GaussianBlur(gray_left, (3, 3), 0)
        gray_right = cv2.GaussianBlur(gray_right, (3, 3), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_left = clahe.apply(gray_left)
        gray_right = clahe.apply(gray_right)
        
        disparity = self.stereo_matcher.compute(gray_left, gray_right)
        
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity

    def calculate_depth(self, disparity_map, detection, baseline, focal_length):
        if detection is None or disparity_map is None:
            return None

        x1, y1, x2, y2 = detection['bbox']

        img_height, img_width = disparity_map.shape[:2]
        pad_x = 20 
        pad_y = 20 
        
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(img_width, x2 + pad_x)
        y2_padded = min(img_height, y2 + pad_y)

        disparity_region = disparity_map[y1_padded:y2_padded, x1_padded:x2_padded] 

        valid_disparities = disparity_region[disparity_region > 0] 

        if len(valid_disparities) == 0:
            print("       [DEBUG] No valid disparities found in detection region.")
            return None

        disparity_value = np.percentile(valid_disparities, 90) 

        if disparity_value < 0.1: 
            print(f"       [DEBUG] Disparity value {disparity_value:.2f} is too small, likely invalid depth.")
            return None
        
        if disparity_value > 0: 
            depth = (baseline * focal_length) / disparity_value
            return depth

        return None

    def visualize_results(self, img_left, img_right, disparity, detection_left, detection_right, depth, wait_for_key=True, display_scale=0.3): 
        """Create visualization using OpenCV (no matplotlib dependency)"""
        # Hardcoded visualization parameters
        text_readability_multiplier = 1.5 
        base_font_scale = 0.5
        base_line_thickness = 1
        wait_key_time = 2000 # milliseconds

        if img_left is not None:
            h, w = img_left.shape[:2]
            img_left_resized = cv2.resize(img_left, (int(w * display_scale), int(h * display_scale)))
        else:
            img_left_resized = np.zeros((int(480 * display_scale), int(640 * display_scale), 3), dtype=np.uint8)

        if img_right is not None:
            h_r, w_r = img_right.shape[:2] 
            img_right_resized = cv2.resize(img_right, (int(w_r * display_scale), int(h_r * display_scale)))
        else:
            img_right_resized = np.zeros((int(480 * display_scale), int(640 * display_scale), 3), dtype=np.uint8)

        display_left = img_left_resized.copy()
        display_right = img_right_resized.copy()
        
        actual_font_scale = base_font_scale * text_readability_multiplier 
        actual_line_thickness = max(1, int(base_line_thickness * text_readability_multiplier))

        # Extract center coordinates once here if detections exist
        left_center_x, left_center_y = None, None
        right_center_x, right_center_y = None, None

        if detection_left:
            x1, y1, x2, y2 = [int(coord * display_scale) for coord in detection_left['bbox']]
            left_center_x, left_center_y = [int(coord * display_scale) for coord in detection_left['center']]
            
            cv2.rectangle(display_left, (x1, y1), (x2, y2), (0, 255, 0), actual_line_thickness)
            cv2.circle(display_left, (left_center_x, left_center_y), int(5 * actual_font_scale), (0, 0, 255), -1) 
            
            depth_text = f"Depth: {depth:.2f}m" if depth is not None else "No depth"
            text_y_pos = y1 - int(10 * actual_font_scale)
            if text_y_pos < 10: 
                text_y_pos = y2 + int(20 * actual_font_scale)

            cv2.putText(display_left, depth_text, (x1, text_y_pos),
                                 cv2.FONT_HERSHEY_SIMPLEX, actual_font_scale, (0, 255, 0), actual_line_thickness)
            cv2.putText(display_left, f"Conf: {detection_left['confidence']:.2f}", (x1, text_y_pos + int(20 * actual_font_scale)),
                                 cv2.FONT_HERSHEY_SIMPLEX, actual_font_scale, (0, 255, 0), actual_line_thickness)
            
            cv2.putText(display_left, f"Y_pos: {detection_left['center'][1]}", (x1, text_y_pos + int(40 * actual_font_scale)),
                                 cv2.FONT_HERSHEY_SIMPLEX, actual_font_scale, (0, 255, 255), actual_line_thickness)


        if detection_right:
            x1_r, y1_r, x2_r, y2_r = [int(coord * display_scale) for coord in detection_right['bbox']]
            right_center_x, right_center_y = [int(coord * display_scale) for coord in detection_right['center']]
            cv2.rectangle(display_right, (x1_r, y1_r), (x2_r, y2_r), (0, 255, 0), actual_line_thickness)
            cv2.putText(display_right, f"Conf: {detection_right['confidence']:.2f}", (x1_r, y1_r - int(10 * actual_font_scale)),
                                 cv2.FONT_HERSHEY_SIMPLEX, actual_font_scale, (0, 255, 0), actual_line_thickness)
            
            cv2.putText(display_right, f"Y_pos: {detection_right['center'][1]}", (x1_r, y1_r - int(30 * actual_font_scale)),
                                 cv2.FONT_HERSHEY_SIMPLEX, actual_font_scale, (0, 255, 255), actual_line_thickness)
            
        disp_to_display = None
        if disparity is not None and disparity.size > 0:
            print(f"       [DEBUG] Disparity map max value: {np.max(disparity):.2f}")
            if np.max(disparity) > 0:
                disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                disp_to_display = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
                disp_to_display = cv2.resize(disp_to_display, (display_left.shape[1], display_left.shape[0]))
            else:
                disp_to_display = np.zeros_like(display_left, dtype=np.uint8) 
                cv2.putText(disp_to_display, "Zero Disparity Map (No Depth)", (int(50 * actual_font_scale), disp_to_display.shape[0] // 2),
                                                 cv2.FONT_HERSHEY_SIMPLEX, actual_font_scale, (255, 255, 255), actual_line_thickness)
        else:
            disp_to_display = np.zeros_like(display_left, dtype=np.uint8) 
            cv2.putText(disp_to_display, "No Disparity Data Available", (int(50 * actual_font_scale), disp_to_display.shape[0] // 2),
                                                 cv2.FONT_HERSHEY_SIMPLEX, actual_font_scale, (255, 255, 255), actual_line_thickness)
        
        status_message = ""
        # Hardcoded tolerance values
        y_tolerance = 300 
        min_x_disparity = 5 

        if detection_left is None and detection_right is None:
            status_message = "Stop sign not detected in either image."
        elif detection_left is None:
            status_message = "Stop sign not detected in left image."
        elif detection_right is None:
            status_message = "Stop sign detected in the left image, but not in the right. Cannot calculate depth."
        # Use the extracted variables here
        elif detection_left and detection_right and abs(left_center_y - right_center_y) > y_tolerance: 
            status_message = (f"Large Y-disparity: {abs(left_center_y - right_center_y):.2f}"
                              f". (Above threshold {y_tolerance}). Stereo calibration is needed!")
        elif detection_left and detection_right and (left_center_x - right_center_x) < min_x_disparity: 
            status_message = (f"Small X-disparity: {left_center_x - right_center_x:.2f}. "
                              f"(Below threshold {min_x_disparity}). Possible mismatch or poor calibration.")
        elif depth is None:
            status_message = "Valid disparity but no depth computed."
        else:
            status_message = f"Depth: {depth:.2f}m"

        cv2.putText(disp_to_display, status_message, (int(50 * actual_font_scale), disp_to_display.shape[0] - int(30 * actual_font_scale)),
                                 cv2.FONT_HERSHEY_SIMPLEX, actual_font_scale, (0, 255, 255), actual_line_thickness)

        window_name_left = "Left Image + Detection"
        window_name_right = "Right Image + Detection"
        window_name_disparity = "Disparity Map & Status" 

        cv2.namedWindow(window_name_left, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_name_right, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_name_disparity, cv2.WINDOW_NORMAL)

        cv2.imshow(window_name_left, display_left)
        cv2.imshow(window_name_right, display_right)
        cv2.imshow(window_name_disparity, disp_to_display)

        default_window_w = display_left.shape[1]
        default_window_h = display_left.shape[0]

        cv2.resizeWindow(window_name_left, default_window_w, default_window_h)
        cv2.resizeWindow(window_name_right, default_window_w, default_window_h)
        cv2.resizeWindow(window_name_disparity, default_window_w, default_window_h) 

        margin_x = 10 
        gap_x = 30 
        gap_y = 30 

        cv2.moveWindow(window_name_left, margin_x, 10) 
        cv2.moveWindow(window_name_right, margin_x + default_window_w + gap_x, 10) 

        total_top_row_width = (default_window_w * 2) + gap_x 
        disp_x_pos = margin_x + (total_top_row_width - default_window_w) // 2 
        disp_y_pos = 10 + default_window_h + gap_y 
        
        cv2.moveWindow(window_name_disparity, disp_x_pos, disp_y_pos)

        if wait_for_key:
            print(f"Press any key to continue...")
            cv2.waitKey(wait_key_time) 
        else:
            cv2.waitKey(wait_key_time) 
        cv2.destroyAllWindows()

    def extract_frames_from_video(self, video_path, output_folder, fps=1):
        """Extracts frames from a video at a specified FPS."""
        # Hardcoded FPS
        print(f"Clearing existing output folder: {output_folder}...")
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0

        frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
        if frame_rate == 0:
            print(f"Error: Could not get frame rate from {video_path}")
            return 0

        # Calculate which frames to skip to achieve the desired fps
        frames_to_skip = int(frame_rate / fps) if fps > 0 else 1
        
        current_frame_index = 0 # Tracks the index of the frame read from the video
        extracted_count = 0     # Tracks how many frames have actually been extracted and saved

        print(f"Extracting frames from {video_path} to {output_folder} at {fps} FPS...")
        should_continue_reading = True # Flag to control the loop

        while should_continue_reading:
            success, image = vidcap.read()

            if not success:
                should_continue_reading = False # Stop loop if reading fails (end of video)
            else:
                # Only save frame if it's at the desired interval
                if current_frame_index % frames_to_skip == 0:
                    frame_filename = os.path.join(output_folder, f"frame_{extracted_count:05d}.jpg")
                    cv2.imwrite(frame_filename, image)
                    extracted_count += 1
                
                current_frame_index += 1 # Increment overall frame counter

        vidcap.release() # Release the video capture object
        print(f"Finished extracting {extracted_count} frames from {video_path}.")
        return extracted_count

    def process_stereo_frames(self, left_frames_folder, right_frames_folder):
        """Processes stereo image pairs from folders for depth calculation."""
        print(f"\n--- Starting Stereo Video Processing ---")
        
        baseline, focal_length = self.extract_calibration_data(None) 
        print(f"Using estimated baseline: {baseline:.3f}m, focal length: {focal_length:.1f}px")

        left_images = sorted(glob.glob(os.path.join(left_frames_folder, "*.jpg")))
        right_images = sorted(glob.glob(os.path.join(right_frames_folder, "*.jpg")))

        print(f"Found {len(left_images)} .jpg files in {left_frames_folder}")
        print(f"Found {len(right_images)} .jpg files in {right_frames_folder}")

        if len(left_images) != len(right_images):
            print("Error: Number of frames in left and right timings do not match.")
            return

        print(f"Found {len(left_images)} stereo frame pairs to process.")

        # Hardcoded tolerance values
        y_tolerance = 300 
        min_x_disparity = 5 

        for idx, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images)):
            print(f"\nProcessing stereo frame pair {idx+1}/{len(left_images)}")

            img_left = cv2.imread(left_img_path)
            img_right = cv2.imread(right_img_path)

            detection_left = None
            detection_right = None
            disparity_map = None
            depth = None
            status_message = ""
            should_wait_for_key = True 

            if img_left is None or img_right is None:
                status_message = "Error: Could not load one or more stereo frames."
                dummy_shape = (int(480 * 0.3), int(640 * 0.3), 3) 
                img_left = np.zeros(dummy_shape, dtype=np.uint8)
                img_right = np.zeros(dummy_shape, dtype=np.uint8)
                disparity_map = np.zeros((dummy_shape[0], dummy_shape[1]), dtype=np.float32) 
                should_wait_for_key = False
            else:
                detection_left = self.detect_stop_sign(img_left)
                detection_right = self.detect_stop_sign(img_right)

                # Try to find detection in both images if not found in one
                if detection_left is None and detection_right is not None:
                    temp_conf = self.confidence_threshold * 0.8 
                    print(f" Re-attempting left detection with lower confidence ({temp_conf:.2f})...")
                    # When not using config, instantiate directly with the confidence
                    temp_processor = StereoYOLOProcessor(confidence_threshold=temp_conf) 
                    detection_left = temp_processor.detect_stop_sign(img_left)
                    if detection_left:
                        print(f" Left detected successfully with lower confidence.")
                
                if detection_right is None and detection_left is not None:
                    temp_conf = self.confidence_threshold * 0.8 
                    print(f" Re-attempting right detection with lower confidence ({temp_conf:.2f})...")
                    # When not using config, instantiate directly with the confidence
                    temp_processor = StereoYOLOProcessor(confidence_threshold=temp_conf)
                    detection_right = temp_processor.detect_stop_sign(img_right)
                    if detection_right:
                        print(f" Right detected successfully with lower confidence.")


                if detection_left is None or detection_right is None:
                    if detection_left is None and detection_right is None:
                        status_message = "Stop sign not detected in either image."
                    elif detection_left is None:
                        status_message = "Stop sign not detected in the left image."
                    else: 
                        status_message = "Stop sign detected in the left image, but not in the right. Cannot calculate depth."
                    should_wait_for_key = False
                else:
                    left_center_x, left_center_y = detection_left['center']
                    right_center_x, right_center_y = detection_right['center']

                    # Ensure X-difference is correctly calculated (left_x - right_x for disparity)
                    x_diff = left_center_x - right_center_x 

                    if abs(left_center_y - right_center_y) > y_tolerance:
                        status_message = (f"Stop sign detected in both, but Y-difference is too large: "
                                          f"{abs(left_center_y - right_center_y):.2f}. (Above threshold {y_tolerance}). "
                                          f"Stereo calibration is needed!")
                        should_wait_for_key = True 
                    elif x_diff < min_x_disparity: 
                        status_message = (f"Stop sign detected in both, but X-difference is too small: "
                                          f"{x_diff:.2f}. (Below threshold {min_x_disparity}). "
                                          f"Possible mismatch or poor calibration.")
                        should_wait_for_key = True 
                    else:
                        disparity_map = self.compute_disparity(img_left, img_right)
                        depth = self.calculate_depth(disparity_map, detection_left, baseline, focal_length)
                        
                        if depth is None:
                            status_message = (f"Stop sign detected (Conf Left: {detection_left['confidence']:.3f}, Right: {detection_right['confidence']:.3f}). "
                                              f"Could not compute valid depth from disparity map.")
                            should_wait_for_key = True 
                        else:
                            status_message = (f"Stop sign detected! Depth: {depth:.2f} meters "
                                              f"(Conf Left: {detection_left['confidence']:.3f}, Right: {detection_right['confidence']:.3f}).")
                            should_wait_for_key = True 
                
            print(f" {status_message}")
            self.visualize_results(img_left, img_right, disparity_map, 
                                   detection_left, detection_right, depth, wait_for_key=should_wait_for_key, display_scale=0.3) 

        print("\n--- Stereo Video Processing Complete ---")

def main_video():
    # Hardcoded values for paths and FPS as per "without config file" request
    processor = StereoYOLOProcessor(model_path="last.pt", confidence_threshold=0.5) 

    left_video_path = "video/LEFT.mp4" 
    right_video_path = "video/RIGHT.mp4" 
    
    left_frames_folder = "extracted_tesla_frames/STOP_LEFT_frames"
    right_frames_folder = "extracted_tesla_frames/STOP_RIGHT_frames"
    extraction_fps = 1 # Hardcoded FPS
    
    print(f"\n--- Starting frame extraction for LEFT Tesla video ---")
    processor.extract_frames_from_video(left_video_path, left_frames_folder, fps=extraction_fps) 
    print(f"\n--- Starting frame extraction for RIGHT Tesla video ---")
    processor.extract_frames_from_video(right_video_path, right_frames_folder, fps=extraction_fps) 

    print(f"\n--- Starting Stop Sign Detection on frames from: {left_frames_folder} (Left Tesla) ---")
    all_left_frames = sorted(glob.glob(os.path.join(left_frames_folder, "*.jpg")))
    print(f"Found {len(all_left_frames)} .jpg files in {left_frames_folder}")
    print(f"Found {len(all_left_frames)} frames to process.")
    for frame_path in all_left_frames:
        print(f"Processing frame: {os.path.basename(frame_path)}")
        img = cv2.imread(frame_path)
        if img is not None:
            detection = processor.detect_stop_sign(img)
            if detection:
                x1, y1, x2, y2 = detection['bbox']
                print(f" Detected Stop Sign: Confidence={detection['confidence']:.2f} at bbox ({x1},{y1},{x2},{y2})")
            else:
                print(" No stop sign detected in this frame.")
        else:
            print(f" Warning: Could not load image {frame_path}")
    print("--- Finished displaying detection results for Left Tesla frames ---")

    print(f"\n--- Starting Stop Sign Detection on frames from: {right_frames_folder} (Right Tesla) ---")
    all_right_frames = sorted(glob.glob(os.path.join(right_frames_folder, "*.jpg")))
    print(f"Found {len(all_right_frames)} .jpg files in {right_frames_folder}")
    print(f"Found {len(all_right_frames)} frames to process.")
    for frame_path in all_right_frames:
        print(f"Processing frame: {os.path.basename(frame_path)}")
        img = cv2.imread(frame_path)
        if img is not None:
            detection = processor.detect_stop_sign(img)
            if detection:
                x1, y1, x2, y2 = detection['bbox']
                print(f" Detected Stop Sign: Confidence={detection['confidence']:.2f} at bbox ({x1},{y1},{x2},{y2})")
            else:
                print(" No stop sign detected in this frame.")
        else:
            print(f" Warning: Could not load image {frame_path}")
    print("--- Finished displaying detection results for Right Tesla frames ---")

    processor.process_stereo_frames(left_frames_folder, right_frames_folder)

    print("\nAll video processing and detection on single frames complete.")

if __name__ == "__main__":
    main_video()