from ultralytics import YOLO
import cv2
import math

# Load the trained model
model = YOLO("runs/detect/train12/weights/best.pt")  # Path to your trained model

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Run inference on the webcam
results = model.predict(
    source=0,  # Use webcam
    conf=0.5,
    show=False,  # Disable YOLO's built-in display
    imgsz=640,
    device=0,
    stream=True  # Use streaming mode for webcam
)

# Loop through the results
for result in results:
    # Get the annotated frame
    frame = result.plot()  # This gives the frame with bounding boxes drawn

    # Get the bounding boxes
    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes in (x1, y1, x2, y2) format

    # List to store centers of all detected objects
    centers = []

    # Calculate centers of all detected objects
    for box in boxes:
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)  # Center of the bounding box
        centers.append(center)

    # Draw lines and calculate distances between all pairs of objects
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            # Get the centers of the two objects
            center1 = centers[i]
            center2 = centers[j]

            # Calculate the distance between the two centers
            distance = calculate_distance(center1, center2)

            # Draw a line between the centers
            cv2.line(frame, (int(center1[0]), int(center1[1])), (int(center2[0]), int(center2[1])), (0, 255, 0), 2)

            # Display the distance on the line
            midpoint = ((int(center1[0]) + int(center2[0])) // 2, (int(center1[1]) + int(center2[1])) // 2)
            cv2.putText(frame, f"{distance:.2f} px", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLO Webcam", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cv2.destroyAllWindows()