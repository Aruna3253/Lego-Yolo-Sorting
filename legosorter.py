import cv2
from detect import detect_objects

# Simulated bins for sorting
bins = {
    0: (50, 400),  # Class 0 → Left
    1: (300, 400), # Class 1 → Center
    2: (550, 400)  # Class 2 → Right
}

def simulate_sorting():
    cap = cv2.VideoCapture(0)  # or 'data/video.mp4' for a recorded video

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            conf = det['conf']
            label = f"Class {cls} ({conf:.2f})"

            # Draw detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # Simulate sorting line by drawing arrows to bins
            bin_x, bin_y = bins.get(cls, (300, 450))
            cv2.line(frame, (int((x1 + x2) / 2), y2), (bin_x, bin_y), (0, 0, 255), 2)

        # Draw bins
        for cls, (bx, by) in bins.items():
            cv2.rectangle(frame, (bx-40, by-40), (bx+40, by+40), (255,0,0), 2)
            cv2.putText(frame, f"Bin {cls}", (bx-30, by+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("LEGO Sorter Simulation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    simulate_sorting()
