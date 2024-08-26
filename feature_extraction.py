import cv2
import numpy as np

def extract_features(imgCanvas):
    """
    Extracts features from the given image by identifying and describing shapes.

    Args:
        imgCanvas (np.array): The image containing the shapes to be analyzed.

    Returns:
        list: A list of dictionaries where each dictionary contains details of a detected shape.
    """
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    
    _, imgThresh = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    
    for contour in contours:
        # Approximate the contour to simplify it
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get the number of vertices
        num_vertices = len(approx)
        
        # Determine shape type
        if num_vertices == 3:
            shape_type = "Triangle"
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                shape_type = "Square"
            else:
                shape_type = "Rectangle"
        elif num_vertices == 5:
            shape_type = "Pentagon"
        elif num_vertices == 6:
            shape_type = "Hexagon"
        else:
            shape_type = "Circle"
        
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Calculate the center of the shape (Spatial Moment / Area Moment)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        # Create a mask for the current shape to find colour
        mask = np.zeros_like(imgGray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        mean_color = cv2.mean(imgCanvas, mask=mask)[:3]  # Mean color in BGR format
        mean_color = tuple(map(int, mean_color))

        shapes.append({
            "shape": shape_type,
            "vertices": num_vertices,
            "area": area,
            "center": (cX, cY),
            "color": mean_color
        })

    return shapes

if __name__ == "__main__":

    imgCanvas = cv2.imread('drawing.png')

    shapes = extract_features(imgCanvas)

    for shape in shapes:
        print(shape)
