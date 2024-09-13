from ultralytics import YOLO
import cv2
import torch
import numpy as np

# Check if GPU is available and set device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv8 model
model = YOLO('yolov8x.pt').to(device)

# Open the video file
video_path = "base.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# Start processing from the 5th frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 5)

# Define the codec and create VideoWriter object using 'mp4v' as it's widely supported
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height)) 

# Initialize counter and list to keep track of counted IDs
person_count_path_1 = 0
counted_ids = set()  # Use a set to keep track of counted IDs
prev_positions = {}  # Dictionary to store previous positions of objects

# Define the class index for "person"
yolo_classes = list(model.names.values())
person_class_id = yolo_classes.index("person")

# Define points to draw line within ROI
roi_line_start = (497, 1033)
roi_line_end = (1380, 1005)
roi_line_y = (roi_line_start[1] + roi_line_end[1]) // 2  # Average y-coordinate for the line

# Define ROI with four specific points
roi_points = np.array([[665, 780], [1220, 780], [1700, 1500], [210, 1500]])

# Function to check if a person has crossed the line
def has_crossed_line(prev_y, curr_y, line_y):
    return (prev_y <= line_y < curr_y) or (prev_y >= line_y > curr_y)

# Read frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fixed size
    # frame = cv2.resize(frame, (frame_width, frame_height))

    # Create a mask for the ROI
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)

    # Extract ROI from the frame using the mask
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Get the bounding rectangle of the ROI for cropping
    x, y, w, h = cv2.boundingRect(roi_points)

    # Crop the ROI from the frame
    cropped_roi_frame = roi_frame[y:y+h, x:x+w]

    # Detect and track only people within the cropped ROI
    results = model.track(cropped_roi_frame, classes=[person_class_id], device=device, persist=True)  # Tracking only class 'person'

    # Draw a polygon around the ROI for visualization
    cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue color in BGR, thickness 2

    # Draw red line within the ROI on the original frame
    cv2.line(frame, (roi_line_start[0], roi_line_start[1]), (roi_line_end[0], roi_line_end[1]), (0, 0, 255), 2)  # Red color in BGR, thickness 2

    # Draw the detection results on the original frame
    for result in results:
        boxes = result.boxes.xyxy.tolist()
        ids = result.boxes.id.tolist() if result.boxes.id is not None else [None] * len(boxes)
        
        # Print boxes and ids
        print(f'Boxes: {boxes}')
        print(f'IDs: {ids}')

        for i in range(len(boxes)):
            if ids[i] is None:
                continue

            x1, y1, x2, y2 = int(boxes[i][0]) + x, int(boxes[i][1]) + y, int(boxes[i][2]) + x, int(boxes[i][3]) + y
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Check if the object has crossed the line
            if ids[i] in prev_positions:
                prev_x, prev_y = prev_positions[ids[i]]
                if has_crossed_line(prev_y, cy, roi_line_y) and ids[i] not in counted_ids:
                    counted_ids.add(ids[i])
                    person_count_path_1 += 1

                # Draw the flow arrow within the bounding box
                arrow_start = (int(prev_x), int(prev_y))
                arrow_end = (int(cx), int(cy))
                arrow_start = (max(x1, min(x2, arrow_start[0])), max(y1, min(y2, arrow_start[1])))
                arrow_end = (max(x1, min(x2, arrow_end[0])), max(y1, min(y2, arrow_end[1])))
                cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 0, 255), 2, tipLength=0.3)  # Red color in BGR

            # Update the previous position
            prev_positions[ids[i]] = (cx, cy)

            # Draw the bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {ids[i]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f'Person Count: {person_count_path_1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Write the frame into the file 'output4.mp4'
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
