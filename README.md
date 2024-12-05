# Accident_Detection
Accident Detection with Deep Learning and Streamlit Deployment

## Overview
This project involves training a deep learning model to classify images as either "Accident" or "Non-Accident." The solution utilizes a convolutional neural network (CNN) built on the VGG16 architecture, combined with visualization tools for edge detection, contour analysis, and heatmaps. The model is deployed using Streamlit for easy interaction and demonstration.

## Features
- **Deep Learning Model**: Uses a VGG16-based architecture for binary classification.
- **Data Preprocessing**: Efficient data loading and augmentation for improved model generalization.
- **Visualization Tools**: Includes edge detection, contour mapping, and heatmap generation for enhanced interpretability.
- **Streamlit Deployment**: Provides a user-friendly web interface for uploading and classifying images.

## Files Included
- **Notebook (`image-and-video.ipynb`)**: Contains the code for training, evaluating, and visualizing the model.
- **Streamlit Application (`app.py`)**: Hosts the model and provides a web interface for end users.

## Installation
To run this project, follow these steps:

### Prerequisites
Ensure you have Python 3.8 or above installed. The following libraries are required:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Streamlit

You can install these dependencies using:
```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python-headless streamlit
```

### Dataset
The project uses a dataset containing images labeled as "Accident" or "Non-Accident." Place the dataset in a folder structure as shown below:
```
data/
  train/
    accident/
    non-accident/
  validation/
    accident/
    non-accident/
```
Update the file paths in the notebook as required.

## Usage

### Running the Notebook
1. Open the notebook in Jupyter or any compatible environment.
2. Execute the cells step-by-step to train the model and visualize the results.

### Deploying the Streamlit App
1. Ensure the trained model is saved in a compatible format (e.g., `.h5`).
2. Update the `app.py` file to point to the saved model file.
3. Run the Streamlit app using:
   ```bash
   streamlit run app.py
   ```
4. Open the link provided in the terminal to access the web application.

### Features of the App
- Upload an image to classify it as "Accident" or "Non-Accident."
- View edge detection, contour mapping, and heatmap visualizations for uploaded images.

## Results
- The trained model achieved an accuracy of ~90% on the validation set.
- Confusion matrix and other metrics are visualized in the notebook.

## Visualization Examples
1. **Edge Detection**: Identifies edges in the input images.
2. **Contour Mapping**: Highlights significant contours in the images.
3. **Heatmap**: Applies a heatmap overlay for feature visualization.

## License
This project is open-source and available under the MIT License. Feel free to modify and use it as needed.

## Acknowledgments
- The VGG16 model was adapted from TensorFlow's pre-trained models.
- Dataset curated from publicly available accident detection datasets.

---
For questions or feedback, please contact [your email or GitHub profile link].


