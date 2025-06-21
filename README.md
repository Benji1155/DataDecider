![DataDecider](https://i.imgur.com/Q2NDchn.png)

# **DataDecider - An AI‑Driven Chatbot for Data Visualization and Statistical Guidance**

[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Benji1155/DataDecider)](https://img.shields.io/github/v/release/Benji1155/DataDecider)
[![GitHub last commit](https://img.shields.io/github/last-commit/Benji1155/DataDecider)](https://img.shields.io/github/last-commit/Benji1155/DataDecider)
[![GitHub issues](https://img.shields.io/github/issues-raw/Benji1155/DataDecider)](https://img.shields.io/github/issues-raw/Benji1155/DataDecider)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/Benji1155/DataDecider)](https://img.shields.io/github/issues-pr/Benji1155/DataDecider)
[![GitHub](https://img.shields.io/github/license/Benji1155/DataDecider)](https://img.shields.io/github/license/Benji1155/DataDecider)

---

## About The Project

Link: https://datadecider.onrender.com/

DataDecider is an intelligent, conversational AI assistant built to bridge the gap between having data and knowing what to do with it. Many students, researchers, and professionals face a common roadblock: they have a dataset and a question, but are unsure which statistical test or visualization is appropriate for their needs. This can lead to wasted time, frustration, and potentially incorrect conclusions.

This project solves that problem by providing a user-friendly, web-based chatbot that guides users to the right analytical methods. It's designed to be accessible to users with any level of statistical knowledge.

### Key Features:

* **🧠 Hybrid Conversational Model:** DataDecider combines two powerful approaches for a seamless user experience:
    * **Guided Assistance:** For users who are unsure where to start, the bot initiates a multi-step conversation, asking targeted questions to understand their goals and data. It then provides tailored recommendations for statistical tests and charts.
    * **Direct Q&A:** For users with specific questions, a custom-trained Natural Language Understanding (NLU) model provides instant, easy-to-understand explanations for over 30 common statistical concepts and chart types.
* **🤖 Auto-Mode:** For users who want a quick overview, this feature performs an automated analysis of an uploaded dataset, instantly providing a data summary, key statistics, and suggested visualizations.
* **📊 Dynamic Chart Generation:** The application uses Matplotlib and Seaborn on the backend to generate and display data visualizations based on user input and recommendations.
* **📚 Educational Focus:** All explanations are written in simple terms with analogies and examples, making complex topics accessible to everyone.

This project was developed as a "live brief" for the ATW306 - Advanced Tech: Work Integrated Learning component at Media Design School.

---

### Built With

This project was built with a full-stack Python and AI-centric approach.

* **Backend:**
    * ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
    * ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
    * ![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
* **Machine Learning & NLU:**
    * ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
    * ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
    * ![NLTK](https://img.shields.io/badge/NLTK-3776AB?style=for-the-badge&logo=nltk&logoColor=white)
* **Data Visualization:**
    * ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)
    * ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=for-the-badge&logo=seaborn&logoColor=white)
* **Deployment:**
    * ![Gunicorn](https://img.shields.io/badge/gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)
    * ![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

---

### Getting Started

To get a local copy up and running, follow these simple steps.

#### Prerequisites

You will need Python 3.9+ and pip installed on your machine.

#### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/Benji1155/DataDecider.git](https://github.com/Benji1155/DataDecider.git)
    cd DataDecider
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Train the NLU Model:**
    Before you can run the application, you must train the model using the provided scripts.
    ```sh
    python train_model.py
    ```
    This will create three essential files: `chatbot_model.h5`, `words.pkl`, and `classes.pkl`.

5.  **Run the Flask Application:**
    ```sh
    flask run
    ```
    The application will now be running on your local machine, typically at `http://127.0.0.1:5000`.

---

### Usage

Once the application is running, you can interact with it in several ways:

* **Ask a general question:** Try typing "What is a t-test?" or "What's the difference between mean and median?"
* **Start a guided flow:** Click one of the suggestion buttons like "Help Select Visualizations" to begin a step-by-step conversation.
* **Upload your data:** Use the "Upload File" button to upload a `.csv` file, then try the "Show data summary" or "Auto-Mode" features.
