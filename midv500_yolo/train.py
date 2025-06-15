# train.py

from ultralytics import YOLO

def main():
    # Path to your dataset YAML file
    data_yaml = '/home/ddp/ocr/id-card-weights/midv500_yolo/dataset.yaml'
    
    # Load the YOLOv8-nano model
    model = YOLO('yolo11n.pt')  # You can also use yolov8s.pt, yolov8m.pt, etc. 

    # Train the model
    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # Change to 'cpu' if no GPU
        name='id_card_yolov8n'
    )

    # Optional: Evaluate on validation set
    metrics = model.val()
    print("Validation metrics:", metrics)

    # Optional: Run a test prediction (inference)
    model.predict(source='/home/ddp/ocr/id-card-weights/midv500_yolo/images/test', save=True)

if __name__ == '__main__':
    main()
