# Backend Setup and Deployment Guide

##  Prerequisites

Ensure that the following dependencies are installed before proceeding:

- **Python 3.8+**
- **pip** (Python package manager)
- **Virtual environment** (Recommended for dependency management)

---

##  Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Xzayogn-ECS/xzayogn-client-backend
cd xzayogn-client-backend
```

### 2️⃣ Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate on macOS/Linux
venv\Scripts\activate  # Activate on Windows
```

### 3️⃣ Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

##  Configuration

### Environment Variables

Create a `.env` file in the `backend` directory and configure it as follows:

```env
API_KEY=your_api_key_here
DATABASE_URL=your_database_url_here
```

---

##  Running the Backend Server

Navigate to the backend directory:

```bash
cd backend
```

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be accessible at:

```
http://127.0.0.1:8000
```

---


