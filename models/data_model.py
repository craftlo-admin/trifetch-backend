"""
Data Models - Model Layer
Defines the data structures used in the application
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PatientEventData(BaseModel):
    """Patient event data model"""
    patient_name: str = Field(..., description="Patient IR ID")
    device: str = Field(..., description="Device name")
    event: str = Field(..., description="Event name (AF, PAUSE, VTACH)")
    event_time: str = Field(..., description="Event occurred time")
    time_in_queue: str = Field(..., description="Time elapsed since event occurred")
    technician: str = Field(..., description="Technician name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_name": "75D3DBDB-36C5-451B-9D57-3720224D20FF",
                "device": "Demo9911",
                "event": "AF",
                "event_time": "2025-11-08 14:09:19.884",
                "time_in_queue": "9 days, 2 hours, 15 minutes",
                "technician": "System Admin"
            }
        }


class DataResponse(BaseModel):
    """Response model for fetch data endpoint"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: List[PatientEventData] = Field(..., description="List of patient event data")
    total_count: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    current_page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
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
                "total_pages": 14,
                "current_page": 1,
                "page_size": 10
            }
        }


class ECGDataPoint(BaseModel):
    """Single ECG data point (comma-separated pair)"""
    value1: int = Field(..., description="First value")
    value2: int = Field(..., description="Second value")


class ECGDataResponse(BaseModel):
    """Response model for ECG data endpoint"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    patient_id: str = Field(..., description="Patient IR ID")
    category: str = Field(..., description="Event category")
    patient_folder: str = Field(..., description="Patient folder name")
    event_name: str = Field(..., description="Event name")
    event_time: str = Field(..., description="Event occurred time")
    ecg_data: List[ECGDataPoint] = Field(..., description="Combined ECG data from all files")
    total_data_points: int = Field(..., description="Total number of ECG data points")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
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
        }
