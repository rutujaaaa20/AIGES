import cv2

# Load video
cap = cv2.VideoCapture("crowd.mp4")

# People detector (HOG-based, simple but fast)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

THRESHOLD = 10  # adjust later

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))

    # Detect people
    boxes, _ = hog.detectMultiScale(frame, winStride=(8,8))

    count = len(boxes)

    # Draw boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    status = "NORMAL"
    color = (0,255,0)

    if count > THRESHOLD:
        status = "ALERT: HIGH CROWD"
        color = (0,0,255)

    cv2.putText(frame, f"People Count: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, status, (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Crowd Monitoring Prototype", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
