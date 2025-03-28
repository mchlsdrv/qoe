IMVCA QoE
==============================

An application-agnostic framework to predict Quality of Experience (QoE) metrics from encrypted traffic for UDP-based instant messaging video call applications (IMVCAs).

# 1. Dataset

[Download WhatsApp dataset](https://drive.google.com/file/d/10KuI1ucqJ9_Ij62UfL0oFXQ-0Gw7Rvu5/view?usp=drive_link)

# 2. Data collection

For detailed instructions on data collection, please see the [Data Collection Guide](src/create_data/README.md).

# 3. directory Setup

- If you are using the provided dataset, download the data_collection folder and place it in the project's root directory.
- If you are working with your own dataset, make sure it follows the same structure as the provided dataset and place it in the root directory.

# 4. Model Training and Testing

To train and evaluate the models, use the [run_model_cv.py](src/models/run_model_cv.py) script. Adjust the main function in the script to meet your specific needs.