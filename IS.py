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
        
        # Draw bounding boxes
        if "predictions" in result:
            for obj in result["predictions"]:
                print(obj)  # Print the structure of obj

                # If obj is a tuple with 4 elements, unpack them (x, y, w, h)
                if isinstance(obj, tuple) and len(obj) == 4:
                    x, y, w, h = obj
                    class_name = "Unknown"  # If tuple doesn't have class info, use a placeholder
                elif isinstance(obj, dict):
                    # If obj is a dictionary, extract values using keys
                    x = obj.get("x", 0)
                    y = obj.get("y", 0)
                    w = obj.get("width", 0)
                    h = obj.get("height", 0)
                    class_name = obj.get("class", "Unknown")  # Use get to avoid key errors
                else:
                    # Handle cases where obj doesn't match the expected formats
                    print("Unexpected obj format:", obj)
                    continue
                
                # Convert center x, y to top-left coordinates
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                
                # Draw bounding box on the image (using OpenCV)
                cv2.rectangle(segmented_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(segmented_image, class_name, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    workflow_id="detect-count-and-visualize",
    video_reference="Pigfarm_video2.mp4",  # Change to video path
    max_fps=5,
    on_prediction=process_frame
)

# Start the pipeline
pipeline.start()
pipeline.join()

# Release resources
plt.close()
