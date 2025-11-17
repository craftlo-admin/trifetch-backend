"""
Data Routes - View/Router Layer
Defines the API endpoints and handles HTTP requests/responses
"""
from fastapi import APIRouter, HTTPException, Query, status
from models.data_model import DataResponse, ECGDataResponse
from controllers.data_controller import data_controller

router = APIRouter()


@router.get("/fetchdata", response_model=DataResponse, status_code=status.HTTP_200_OK)
async def fetch_data(
    limit: int = Query(default=10, description="Number of records to return (10, 25, or 50)", ge=1, le=50),
    offset: int = Query(default=0, description="Number of records to skip", ge=0)
):
    """
    Fetch patient event data from test-trifetch folder with pagination
    
    Args:
        limit: Number of records to return (default: 10, options: 10, 25, 50)
        offset: Number of records to skip (default: 0)
    
    Returns:
        DataResponse: JSON response containing paginated patient event data
        
    Example:
        GET /api/fetchdata?limit=10&offset=0
        GET /api/fetchdata?limit=25&offset=75
        GET /api/fetchdata?limit=50&offset=100
        
    Response:
        {
            "success": true,
            "message": "Data fetched successfully",
            "data": [
                {
                    "patient_name": "75D3DBDB-36C5-451B-9D57-3720224D20FF",
                    "device": "Demo9911",
                    "event": "AF",
                    "event_time": "2025-11-08 14:09:19.884",
                    "time_in_queue": "9",
                    "technician": "System Admin"
                }
            ],
            "total_count": 138,
            "total_pages": 6,
            "current_page": 4,
            "page_size": 25
        }
    """
    try:
        result = data_controller.fetch_data(limit=limit, offset=offset)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.message
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get("/fetchdata/{patient_id}", response_model=ECGDataResponse, status_code=status.HTTP_200_OK)
async def fetch_patient_ecg_data(patient_id: str):
    """
    Fetch ECG data for a specific patient by Patient_IR_ID
    
    Args:
        patient_id: Patient IR ID (UUID format)
    
    Returns:
        ECGDataResponse: JSON response containing combined ECG data from all files
        
    Example:
        GET /api/fetchdata/75D3DBDB-36C5-451B-9D57-3720224D20FF
        
    Response:
        {
            "success": true,
            "message": "ECG data retrieved successfully",
            "patient_id": "75D3DBDB-36C5-451B-9D57-3720224D20FF",
            "category": "AF_Approved",
            "patient_folder": "74003321",
            "event_name": "AF",
            "event_time": "2025-11-08 14:09:19.884",
            "ecg_data": [
                {"value1": 1447, "value2": 1459},
                {"value1": 1478, "value2": 1520}
            ],
            "total_data_points": 18000
        }
    """
    try:
        result = data_controller.get_patient_ecg_data(patient_id)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND if "not found" in result.message.lower() else status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.message
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
