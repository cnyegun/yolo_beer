from ultralytics import YOLO
import torch


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolo26s.pt")

    model.train(
        data="data.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        device=device,
        patience=30,
        project="results",
        name="beer_model",
        exist_ok=True,
        dropout=0.1,
        cos_lr=True,
    )

    print("Training complete!")
    print("Model saved to: results/beer_model/weights/best.pt")


if __name__ == "__main__":
    main()
