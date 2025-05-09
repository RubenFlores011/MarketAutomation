from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("C:/Users/ruben/Documents/VisulaStudio codes/WarframeMarketAutomation/Version2/trainedModel/weights/best.pt")

# Define input source (image path or 'screen' for live screen capture)
source = "screen"

# Run inference with confidence threshold set to 0.4 and image size set to (3440, 1440)
results = model(source, conf=0.4, imgsz=(3440, 1440))  # Keep the original image size

# Optionally, show the results
results.show()  # This will display the annotated results on the screen
