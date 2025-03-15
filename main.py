import os
import cv2
import time
import math
import datetime


from ultralytics import YOLO
from pymavlink import mavutil
import config  # Importing the config file
 

# Load the YOLO model using the model path from config
try:
    model = YOLO(config.MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model from {config.MODEL_PATH}: {e}")
    exit(1)

# # Connect to flight controller
# master = mavutil.mavlink_connection(
#     device='/dev/serial0',
#     baud=921600,
#     source_system=1 # RPi's system ID
# )


# Function to create a tracker based on the selected type
def create_tracker(tracker_type: str):
    """Create and return a tracker based on the selected tracker type."""
    tracker_types = {
        "BOOSTING": cv2.legacy.TrackerBoosting_create,
        "MIL": cv2.legacy.TrackerMIL_create,
        "KCF": cv2.legacy.TrackerKCF_create,
        "TLD": cv2.legacy.TrackerTLD_create,
        "MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create,
        "MOSSE": cv2.legacy.TrackerMOSSE_create,
        "CSRT": cv2.legacy.TrackerCSRT_create,
    }

    try:
        return tracker_types[tracker_type]()
    except KeyError:
        print(f"Error: Unknown tracker type: {tracker_type}")
        raise ValueError(f"Unknown tracker type: {tracker_type}")

def display_info(frame, frame_width, frame_height, current_fps, object_width, object_height, distance, hor_angle, ver_angle):
    """Display various information on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (180, 190), config.INFO_RECT_COLOR, -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    info = [
        f"Vid Res: {frame_width}x{frame_height} px",
        f"Training Res: {config.TRAINING_RES} px",
        f"Object Width: {object_width} px",
        f"Object Height: {object_height} px",
        f"Distance: {distance:.3f} m",
        f"Hor Angle: {hor_angle} deg",
        f"Ver Angle: {ver_angle} deg",
        current_fps
    ]

    y_offset = 30
    for line in info:
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.INFO_COLOR, 2)
        y_offset += 20

def distance_approx(frame_width, object_width):
    """Approximate the distance to an object based on its width in the frame."""
    if object_width > 0:
        return round(0.3 / ((object_width / frame_width) * math.pi), 2)
    return 0

def calculate_angle(frame, x1, y1, x2, y2, frame_width, frame_height):
    """Calculate the horizontal and vertical angles from the center of the frame."""
    obj_centroid_x = (x1 + x2) // 2
    obj_centroid_y = (y1 + y2) // 2
    ref_point_x = frame_width // 2
    ref_point_y = frame_height // 2

    hor_angle = round((obj_centroid_x - ref_point_x) / ref_point_x * config.FOV / 2)
    ver_angle = round((ref_point_y - obj_centroid_y) / ref_point_y * config.FOV / 2)

    return hor_angle, ver_angle


def adjust_dimensions(frame_width, frame_height):
    """Resize or crop the frame to the specified maximum width and height."""
    if config.CROP_CENTER == True:
        return config.CROP_CENTER_WIDTH, config.CROP_CENTER_HEIGHT
    
    if frame_width > config.MAX_ALLOWED_RES:
        scale_ratio = config.MAX_ALLOWED_RES / frame_width
        frame_width = config.MAX_ALLOWED_RES
        frame_height = int(frame_height * scale_ratio)
    
    if frame_height > config.MAX_ALLOWED_RES:
        scale_ratio = config.MAX_ALLOWED_RES / frame_height
        frame_height = config.MAX_ALLOWED_RES
        frame_width = int(frame_width * scale_ratio)

    return frame_width, frame_height

def crop_center(frame, target_width, target_height):
    """Crop the center of the frame to the specified target dimensions."""
    frame_height, frame_width = frame.shape[:2]
    
    if frame_height < target_height:
        print(f"Error: Target Crop Height is greater than Frame Height")
    
    if frame_width < target_width:
        print(f"Error: Target Crop Width is greater than Frame Width")

    # Calculate the center of the frame
    center_x, center_y = frame_width // 2, frame_height // 2
    
    # Calculate the cropping box
    crop_x1 = max(center_x - target_width // 2, 0)
    crop_y1 = max(center_y - target_height // 2, 0)
    crop_x2 = min(center_x + target_width // 2, frame_width)
    crop_y2 = min(center_y + target_height // 2, frame_height)
    
    # Crop the frame to the center
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return cropped_frame, target_width, target_height



def process_video(Tracker_type: str, video_path: str, output_path: str):
    """Process video for detection and tracking."""
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get Video Attributes
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS))

    # Check if input video resolution is greater than MAX_ALLOWED_RES
    if frame_width > config.MAX_ALLOWED_RES or frame_height > config.MAX_ALLOWED_RES:
        config.RESIZE_FLAG = True
        frame_width, frame_height = adjust_dimensions(frame_width, frame_height)

    if config.SAVE_OUTPUT_VIDEO:
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps_input, (frame_width, frame_height))

    tracker = None
    bbox = None
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        

        # check if resizing/cropping is required
        if config.CROP_CENTER == True:
            frame, frame_width, frame_height = crop_center(frame, config.CROP_CENTER_WIDTH, config.CROP_CENTER_HEIGHT)
        elif config.RESIZE_FLAG == True:
            frame = cv2.resize(frame, (frame_width, frame_height))


        if frame_count % config.DETECTION_INTERVAL == 0 or tracker is None or not bbox:
            try:
                # Run YOLO detection
                results = model.predict(frame, stream=True, verbose=False)
            except Exception as e:
                print(f"Error during YOLO detection: {e}")
                continue

            bbox = None
            for result in results:
                boxes = result.boxes.cpu().numpy()
                if boxes:
                    for box in boxes:
                        class_name = "Detection : quadcopter-300mm"
                        confidence = float(box.conf[0])       # Confidence score
                        x1, y1, w, h = map(int, box.xywh[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        break  # Track only the first detected object
                    
                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Prepare text with class name and confidence score
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                    
                    hor_angle, ver_angle = calculate_angle(frame, x1, y1, x2, y2, frame_width, frame_height)

                else:
                    w = h = hor_angle = ver_angle = 0

            if bbox:
                try:
                    # Initialize the tracker
                    tracker = create_tracker(Tracker_type)
                    tracker.init(frame, bbox)
                except ValueError as e:
                    print(f"Error initializing tracker: {e}")
                    return

        else:
            # Update the tracker
            success, bbox = tracker.update(frame)
            if success:
                x1, y1, w, h = map(int, bbox)
                x2, y2 = x1 + w, y1 + h
                # Draw bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Tracking : quadcopter-300mm"
                cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                hor_angle, ver_angle = calculate_angle(frame, x1, y1, x2, y2, frame_width, frame_height)
            else:
                print("Tracker failed! Re-detecting...")
                bbox = None
                tracker = None
                w = h = hor_angle = ver_angle = 0

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        current_fps = f"FPS: {fps:.2f}"

        distance = distance_approx(frame_width, w)

        # Update Parameters

        params = {

            "DISTANCE": distance,   
            "HORIZONTAL ANGLE": hor_angle, 
            "VERTICAL ANGLE": ver_angle      
        }

        # for param_id, value in params.items():
        #     master.mav.param_set_send(
        #         master.target_system,
        #         master.target_component,
        #         param_id.encode(),
        #         value,
        #         mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        #     )
        #     ack = master.recv_match(type='PARAM_VALUE', timeout=5)
        #     if not ack:
        #         print(f"Failed to set {param_id}")

        print(params)

        display_info(frame, frame_width, frame_height, current_fps, w, h, distance, hor_angle, ver_angle)

        if config.SAVE_OUTPUT_VIDEO:
            # Write the frame to the output video
            out.write(frame)

        if config.VISUALIZE_FRAME == True:
            cv2.imshow('Detection and Tracking', frame)

        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release resources
    cap.release()
    
    if config.SAVE_OUTPUT_VIDEO:
        out.release()

    cv2.destroyAllWindows()
    print(f"Output video saved as {output_path}")

def main():


    current_date_time = datetime.datetime.now()
    video_name = current_date_time.strftime("%Y-%m-%d--%H-%M-%S")   # Formatted DateTime
    output_path = config.OUTPUT_PATH_TEMPLATE.format(tracker_type=config.TRACKER_TYPE, video_name=video_name)

    if not os.path.exists(os.path.dirname(output_path)):
        print(f'''Output directory "{os.path.dirname(output_path)}" does not exist! \n creating it now...!''')
        os.makedirs(os.path.dirname(output_path))

    process_video(config.TRACKER_TYPE, config.VIDEO_PATH, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
