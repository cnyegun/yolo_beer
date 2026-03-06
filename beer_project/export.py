from ultralytics import YOLO
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python export.py <format>")
        print("Formats: onnx, engine, torchscript")
        return

    format_type = sys.argv[1]
    model = YOLO("results/beer_model/weights/best.pt")

    model.export(format=format_type)
    print(f"Exported to {format_type} format")


if __name__ == "__main__":
    main()
