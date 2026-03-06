from ultralytics import YOLO
import cv2
import sys


def predict_image(model, path):
    results = model(path)
    results[0].show()
    print(f"Detected {len(results[0].boxes)} objects")
    for box in results[0].boxes:
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        print(f"  {name}: {conf:.2%}")


def predict_video(model, source):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"Error: Cannot open {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        cv2.imshow("Detection", results[0].plot())

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_or_video_path>")
        print("       python predict.py 0  (for webcam)")
        return

    model = YOLO("results/beer_model/weights/best.pt")
    source = sys.argv[1]

    if source.endswith((".mp4", ".avi", ".mov")) or source.isdigit():
        predict_video(model, source)
    else:
        predict_image(model, source)


if __name__ == "__main__":
    main()
