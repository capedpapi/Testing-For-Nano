import cv2

# RTSP URL with credentials
rtsp_url = "rtsp://admin:123456@169.254.104.10:554/live"

# Open RTSP stream
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Unable to open RTSP stream")
    exit()

print("✅ Connected to camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received. Retrying...")
        continue

    cv2.imshow("Camera Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
