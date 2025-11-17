"""
Data Controller - Controller Layer
Contains the business logic for data operations
"""
from models.data_model import PatientEventData, DataResponse
from typing import List
from datetime import datetime
from pathlib import Path
import json
import os


class DataController:
    """Controller class for handling data operations"""
    
    def __init__(self):
        # Path to the test-trifetch folder
        self.base_path = Path(__file__).parent.parent / "test-trifetch"
        self.device_name = "Demo9911"
        self.technician_name = "System Admin"
    
    def calculate_time_in_queue(self, event_time_str: str) -> str:
        """
        Calculate time difference between event time and current time
        
        Args:
            event_time_str: Event occurred time as string
            
        Returns:
            Number of days as string
        """
        try:
            # Parse event time (format: "2025-11-08 14:09:19.884")
            event_time = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S.%f")
            
            # Get current time
            current_time = datetime.now()
            
            # Calculate difference
            time_diff = current_time - event_time
            
            # Return only days
            days = time_diff.days
            
            return f"{days}"
        except Exception as e:
            return f"Error calculating time: {str(e)}"
    
    def fetch_data(self, limit: int = 10, offset: int = 0) -> DataResponse:
        """
        Fetch patient event data from test-trifetch folder with pagination
        
        Args:
            limit: Number of records to return (10, 25, or 50)
            offset: Number of records to skip
        
        Returns:
            DataResponse: Response containing paginated patient event data
        """
        try:
            patient_events = []
            
            # Check if base path exists
            if not self.base_path.exists():
                return DataResponse(
                    success=False,
                    message=f"Data folder not found: {self.base_path}",
                    data=[],
                    total_count=0,
                    total_pages=0,
                    current_page=1,
                    page_size=limit
                )
            
            # Iterate through category folders (AF_Approved, PAUSE_Approved, etc.)
            for category_folder in self.base_path.iterdir():
                if not category_folder.is_dir():
                    continue
                
                # Iterate through patient ID folders (74003321, 74071903, etc.)
                for patient_folder in category_folder.iterdir():
                    if not patient_folder.is_dir():
                        continue
                    
                    # Look for event JSON file
                    patient_id = patient_folder.name
                    event_file = patient_folder / f"event_{patient_id}.json"
                    
                    if event_file.exists():
                        try:
                            # Read and parse JSON file
                            with open(event_file, 'r', encoding='utf-8') as f:
                                event_data = json.load(f)
                            
                            # Extract data and create PatientEventData object
                            patient_event = PatientEventData(
                                patient_name=event_data.get("Patient_IR_ID", "Unknown"),
                                device=self.device_name,
                                event=event_data.get("Event_Name", "Unknown"),
                                event_time=event_data.get("EventOccuredTime", "Unknown"),
                                time_in_queue=self.calculate_time_in_queue(
                                    event_data.get("EventOccuredTime", "")
                                ),
                                technician=self.technician_name
                            )
                            
                            patient_events.append(patient_event)
                        except json.JSONDecodeError as je:
                            print(f"Error parsing JSON file {event_file}: {je}")
                        except Exception as e:
                            print(f"Error processing {event_file}: {e}")
            
            # Sort by event time (most recent first)
            patient_events.sort(
                key=lambda x: datetime.strptime(x.event_time, "%Y-%m-%d %H:%M:%S.%f"),
                reverse=True
            )
            
            # Calculate pagination
            total_count = len(patient_events)
            total_pages = (total_count + limit - 1) // limit  # Ceiling division
            current_page = (offset // limit) + 1
            
            # Get paginated data using offset
            start_index = offset
            end_index = offset + limit
            paginated_data = patient_events[start_index:end_index]
            
            return DataResponse(
                success=True,
                message="Data fetched successfully",
                data=paginated_data,
                total_count=total_count,
                total_pages=total_pages,
                current_page=current_page,
                page_size=limit
            )
        except Exception as e:
            # Handle errors appropriately
            return DataResponse(
                success=False,
                message=f"Error fetching data: {str(e)}",
                data=[],
                total_count=0,
                total_pages=0,
                current_page=1,
                page_size=limit
            )
    
    def get_patient_ecg_data(self, patient_ir_id: str):
        """
        Get ECG data for a specific patient by Patient_IR_ID
        
        Args:
            patient_ir_id: Patient IR ID (UUID)
        
        Returns:
            ECGDataResponse: Response containing combined ECG data
        """
        from models.data_model import ECGDataResponse, ECGDataPoint
        
        try:
            # Load all_events.json
            all_events_file = self.base_path / "all_events.json"
            if not all_events_file.exists():
                return ECGDataResponse(
                    success=False,
                    message="all_events.json not found. Run scripts/build_event_index.py first.",
                    patient_id=patient_ir_id,
                    category="",
                    patient_folder="",
                    event_name="",
                    event_time="",
                    ecg_data=[],
                    total_data_points=0
                )
            
            with open(all_events_file, 'r', encoding='utf-8') as f:
                all_events = json.load(f)
            
            # Find the patient by Patient_IR_ID
            patient_event = None
            for event in all_events:
                if event.get("Patient_IR_ID") == patient_ir_id:
                    patient_event = event
                    break
            
            if not patient_event:
                return ECGDataResponse(
                    success=False,
                    message=f"Patient with ID {patient_ir_id} not found",
                    patient_id=patient_ir_id,
                    category="",
                    patient_folder="",
                    event_name="",
                    event_time="",
                    ecg_data=[],
                    total_data_points=0
                )
            
            # Get patient folder path
            category = patient_event.get("category", "")
            patient_folder = patient_event.get("patient_folder", "")
            patient_path = self.base_path / category / patient_folder
            
            if not patient_path.exists():
                return ECGDataResponse(
                    success=False,
                    message=f"Patient folder not found: {patient_path}",
                    patient_id=patient_ir_id,
                    category=category,
                    patient_folder=patient_folder,
                    event_name=patient_event.get("Event_Name", ""),
                    event_time=patient_event.get("EventOccuredTime", ""),
                    ecg_data=[],
                    total_data_points=0
                )
            
            # Find all ECG data files (ECGData_*.txt)
            ecg_files = sorted(patient_path.glob("ECGData_*.txt"))
            
            if not ecg_files:
                return ECGDataResponse(
                    success=False,
                    message=f"No ECG data files found in {patient_path}",
                    patient_id=patient_ir_id,
                    category=category,
                    patient_folder=patient_folder,
                    event_name=patient_event.get("Event_Name", ""),
                    event_time=patient_event.get("EventOccuredTime", ""),
                    ecg_data=[],
                    total_data_points=0
                )
            
            # Read and combine all ECG files
            ecg_data = []
            for ecg_file in ecg_files:
                try:
                    with open(ecg_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and ',' in line:
                                parts = line.split(',')
                                if len(parts) == 2:
                                    try:
                                        val1 = int(parts[0].strip())
                                        val2 = int(parts[1].strip())
                                        ecg_data.append(ECGDataPoint(value1=val1, value2=val2))
                                    except ValueError:
                                        # Skip invalid lines
                                        continue
                except Exception as e:
                    print(f"Error reading {ecg_file}: {e}")
                    continue
            
            return ECGDataResponse(
                success=True,
                message="ECG data retrieved successfully",
                patient_id=patient_ir_id,
                category=category,
                patient_folder=patient_folder,
                event_name=patient_event.get("Event_Name", ""),
                event_time=patient_event.get("EventOccuredTime", ""),
                ecg_data=ecg_data,
                total_data_points=len(ecg_data)
            )
        
        except Exception as e:
            return ECGDataResponse(
                success=False,
                message=f"Error fetching ECG data: {str(e)}",
                patient_id=patient_ir_id,
                category="",
                patient_folder="",
                event_name="",
                event_time="",
                ecg_data=[],
                total_data_points=0
            )


# Create a singleton instance
data_controller = DataController()
