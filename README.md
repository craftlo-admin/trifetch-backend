# FastAPI MVC Application

A FastAPI application following the MVC (Model-View-Controller) pattern.

## Project Structure

```
Backend/
├── main.py                 # Application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── models/               # Model layer (data structures)
│   ├── __init__.py
│   └── data_model.py
├── controllers/          # Controller layer (business logic)
│   ├── __init__.py
│   └── data_controller.py
└── routes/              # View/Router layer (API endpoints)
    ├── __init__.py
    └── data_routes.py
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file (optional):
```bash
cp .env.example .env
```

## Running the Application

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload
```

The application will be available at: `http://localhost:8000`

## API Documentation

Once the application is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Fetch Data
- **Endpoint**: `GET /api/fetchdata`
- **Description**: Fetches all patient event data from the test-trifetch folder
- **Data Source**: Reads event JSON files from categorized folders (AF_Approved, AF_Rejected, PAUSE_Approved, PAUSE_Rejected, VTACH_Approved, VTACH_Rejected)
- **Response**:
```json
{
    "success": true,
    "message": "Data fetched successfully",
    "data": [
        {
            "patient_name": "75D3DBDB-36C5-451B-9D57-3720224D20FF",
            "device": "Demo9911",
            "event": "AF",
            "event_time": "2025-11-08 14:09:19.884",
            "time_in_queue": "8 days, 11 hours, 50 minutes",
            "technician": "System Admin"
        }
    ],
    "total_count": 138
}
```

**Response Fields:**
- `patient_name`: Patient IR ID from the event JSON file
- `device`: Always "Demo9911"
- `event`: Event type (AF, PAUSE, or VTACH)
- `event_time`: Timestamp when the event occurred
- `time_in_queue`: Calculated time elapsed from event occurrence to current time
- `technician`: Always "System Admin"

## MVC Architecture

- **Model** (`models/`): Defines data structures using Pydantic models
- **Controller** (`controllers/`): Contains business logic and data operations
- **View/Router** (`routes/`): Handles HTTP requests and responses

## Features

- ✅ MVC architecture pattern
- ✅ FastAPI framework
- ✅ Pydantic data validation
- ✅ CORS middleware enabled
- ✅ Auto-generated API documentation
- ✅ Environment-based configuration
- ✅ Type hints and documentation
- ✅ Dynamic file system traversal for data collection
- ✅ Automatic time-in-queue calculation
- ✅ Handles multiple event categories (AF, PAUSE, VTACH)
- ✅ Processes approved and rejected events

## Data Structure

The application reads from the `test-trifetch` folder with the following structure:
```
test-trifetch/
├── AF_Approved/
│   └── {patient_id}/
│       ├── ECGData_200_*.txt (3 files)
│       └── event_{patient_id}.json
├── AF_Rejected/
├── PAUSE_Approved/
├── PAUSE_Rejected/
├── VTACH_Approved/
└── VTACH_Rejected/
```

Each event JSON file contains:
```json
{
    "Patient_IR_ID": "75D3DBDB-36C5-451B-9D57-3720224D20FF",
    "EventOccuredTime": "2025-11-08 14:09:19.884",
    "Event_Name": "AF",
    "IsRejected": "0",
    "EventIndex": 3976
}
```

**Total Events**: 138 (46 AF events, 46 PAUSE events, 46 VTACH events)
