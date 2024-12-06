## EXCEL OCR ADD-IN
This project is an OCR (Optical Character Recognition) Add-in for excel that alllows users to extract text from images and documents. It utilizes various lbraries and frameworks to provide a seamless experince for user. 


![EXCEL OCR Add-in](https://github.com/user-attachments/assets/2afc9f79-c298-4221-8da9-3c3b73e4bbca)


## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic OCR](#basic-ocr)
  - [Table Extraction](#table-extraction)
  - [Posting Results to Excel](#posting-results-to-excel)
  - [Reset Functionality](#reset-functionaity)
- [Requirements](#requirements)
- [License](#license)

## Features
- Extract text from images and PDF documents.
- Supports multiple languages for OCR
- Add extracted data directly into Excel.
- User-friendly interface with drag and drop functionality.
- Post OCR results directly to Excel.
- Translation of extracted text into different languages.


## Installation 
To set up the project, follow these steps. 
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/excel-ocr-addin.git
   cd excel-ocr-addin
   ```

2. **Create a virtual environment (optional but recommeded):**
   ```bash
   python -m venv venv
   source venv/bin/activate # On indows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Download necessary models:**
   The application will automatically download the required model when you run it for the first time. Ensure you have an internet connection. 


## NGROK Integration(optional, you can also use production server for deployement)
To expose your local Flask application to the internet using NGROK, follow these steps:

1. **Download and Install NGROK**

   - Go to the [NGROK website](https://ngrok.com/download) and download the appropriate version for your operating system.
   - Unzip teh downloaded file and place the `ngrok` executable in a directory included in your system's PATH.

2. Authenticate your NGROK account
   - Sign up for a free account on NGROK website
   - After signing in, you will find your authentication token in the dashboard.
   - Run the following command in your terminal to authenticate:
          ```bash
               ngrok authtoken YOUR_AUTH_TOKEN ```
3. **Expose your Flask app**
   - Start your Flask application
     ```bash python app.py```

   - In a new terminal window, run the following comand to expose your app:
     ```bash ngrok http 5000```

   - NGROK will provide you with a public URL that you can use to access your appication from anywhere.
     
## Usage

1. **Run the application through the terminal**
   ```bash
   python app.py
   ```
2. **Access the application**
   Open your web browser and navigate to `http://127.0.0.1:5000` to access the OCR Add-in.

3. **Upload an image or document**
   Drag and drop your image or document into the designated area or click to select a file.

4. **Perform OCR**
  Click on the "Perform OCR" button to extract text from the uploaded file.
  Or use "Select Area" button to select an area from the upload preview and press "Perform OCR" button to extract data from the uplaodd file.

5. **Translate(optional)**
  If you want to translate the extracted text, select the source and the target languages and click the "Translate" button.


### Table Extraction

1. **Select the area for table extraction:**
   After performing OCR, click on the "Extract Table" button to initiate table extraction from the recognized text.

2. **View extracted tables**
  The extracted tables will be displayed in the results section. You can review the tables and make any necessary adjustments.

3. **Post extracted tables to Excel**
  After reviewing, you can post the extracted tables to Excel sheet by clicking the "COPY TABLE" button, the select the starting cell you wanted the table to be posted. From there, just paste it by using Ctrl+V or by right click action.

## Posting Results to Excel
1. **Select an area in Excel**
   Click the "Select Area in Excel" button to choose where you want to post the OCR results. This will highlight the selected range in your Excel workbook.
   
2. **Post OCR Results**
   After performing OCR and selecting the area, click the "POST OCR Results" button.
   The recognized text will be inserted into the selected Excel cells.

3. **Post Translation Results**
  If you have translated the text, you can also post the translation results to Excel by clicking "POST Translation Results" button.

4. **Clear contents of cells**
   To clear contents in cells at Excel directly from this add-in, select an area by using "Select an area in Excel" and then click the "RESET" button.
    
## Clear Functionality
1. **Reset the application state**
  Click the "Clear" button to clear all inputs, results, and selections. This will reset the image preview, OCR results, and any selected areas in Excel.

2. **Clear all fields**
  The reset function will also clear any text in the result and translation result divs, and reset progress bars.

## Requirements
Make sure you have the following installed:
- Python 3.7 or higher
- Flask
- Other dependencies listed in `requirements.txt`




  
