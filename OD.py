import sys
sys.path.append('path_to_the_folder_containing_inference')
from inference import InferencePipeline
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a function to process each frame
def process_frame(result, video_frame):
    if "output_image" in result:
        # Convert the output image to OpenCV format
        segmented_image = result["output_image"].numpy_image
        
        # Draw bounding boxes for detected objects
        if "predictions" in result:
            for obj in result["predictions"]:
                print(obj)  # Print the structure of obj for debugging
                
                # Check if obj is a dictionary containing bounding box and class information
                if isinstance(obj, dict):
                    x = obj.get("x", 0)
                    y = obj.get("y", 0)
                    w = obj.get("width", 0)
                    h = obj.get("height", 0)
                    class_name = obj.get("class", "Unknown")  # Default class name if not present
                    
                    # Convert center x, y to top-left coordinates
                    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                    
                    # Draw bounding box and label
                    cv2.rectangle(segmented_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                    cv2.putText(segmented_image, class_name, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Class label

        # Convert the image from BGR (OpenCV format) to RGB (Matplotlib format)
        segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        
        # Plot the image with Matplotlib
        plt.imshow(segmented_image_rgb)
        plt.axis('off')  # Turn off axes for better display
        plt.title("Segmented Video with Bounding Boxes")
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot

# Initialize the inference pipeline
pipeline = InferencePipeline.init_with_workflow(
    api_key="F7HvX1LuenwkiBjH1NFv",
    workspace_name="baitech",
    workflow_id="detect-count-and-visualize-2",  # Ensure to use the correct workflow ID for your OD model
    video_reference="Pigfarm_video2.mp4",  # Replace with your video path or stream
    max_fps=30,
    on_prediction=process_frame  # This function will process each frame
)

# Start the pipeline
pipeline.start()

# Wait for the pipeline to finish
pipeline.join()

# Release resources
plt.close()
