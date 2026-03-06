from ultralytics import YOLO


def main():
    model = YOLO("results/beer_model/weights/best.pt")

    metrics = model.val(data="dataset/data.yaml")

    print(f"\nResults:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    print(f"\nPer-class:")
    for i, name in model.names.items():
        print(f"  {name}: {metrics.box.ap50[i]:.3f}")


if __name__ == "__main__":
    main()
