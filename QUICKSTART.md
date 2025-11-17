# Quick Start Guide

## Installation & Running

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

Required packages:
- fastapi==0.104.1
- uvicorn==0.24.0
- pydantic==2.5.0
- python-dotenv==1.0.0

### 2. Run the Application

```powershell
python main.py
```

Or using uvicorn directly:

```powershell
uvicorn main:app --reload
```

The server will start at: **http://localhost:8000**

### 3. Access the API

#### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### API Endpoint
```
GET http://localhost:8000/api/fetchdata
```

#### Using curl (PowerShell)
```powershell
curl http://localhost:8000/api/fetchdata
```

#### Using Python requests
```python
import requests

response = requests.get("http://localhost:8000/api/fetchdata")
data = response.json()

print(f"Total events: {data['total_count']}")
for event in data['data'][:5]:
    print(f"{event['event']}: {event['patient_name'][:20]}... - {event['time_in_queue']}")
```

### 4. Expected Response

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

### 5. Testing

Run the test script to verify the controller:

```powershell
python test_controller.py
```

This will display:
- Total records found
- Sample records from each category
- Event distribution statistics

## Troubleshooting

### Port Already in Use
If port 8000 is already in use, specify a different port:

```powershell
uvicorn main:app --reload --port 8001
```

### Module Not Found
Ensure you're in the Backend directory:

```powershell
cd "c:\Users\Himanshu Barnawal\Desktop\Trifetch\Backend"
```

### No Data Returned
Verify that the `test-trifetch` folder exists in the Backend directory with the proper structure.

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/fetchdata` | Fetch all patient events |

## Data Statistics

- **Total Events**: 138
- **Event Types**: AF, PAUSE, VTACH
- **Categories**: Approved and Rejected for each type
- **Average Time in Queue**: ~8-9 days
