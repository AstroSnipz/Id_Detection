# Government ID Classification and Information Extraction

This application uses a trained YOLO model to classify government ID types and extract relevant information. Follow the steps below to set up and run the application.

## Prerequisites

Ensure you have Python installed on your system. If not, download and install it from [python.org](https://www.python.org/).

## Setup Instructions

### Step 1: Clone the Repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/AstroSnipz/Id_Detection.git
cd Id_Detection
```

### Step 2: Create and Activate a Virtual Environment

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

Start the Flask application:

```bash
python app2.py
```

### Step 5: Upload a Government ID

Use the Postman to upload a government ID. The system will classify the ID type.

---

## Troubleshooting

- Ensure all dependencies are correctly installed by re-running `pip install -r requirements.txt`.
- Verify that the `MODEL_PATH` is correct and points to your trained YOLO model.

---

Happy coding!
