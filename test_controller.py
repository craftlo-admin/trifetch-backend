"""
Test script to verify the data controller logic
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from controllers.data_controller import data_controller

# Test the fetch_data method
result = data_controller.fetch_data()

print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Total Count: {result.total_count}")
print(f"\nFirst 5 records:")
print("-" * 80)

for i, event in enumerate(result.data[:5]):
    print(f"\nRecord {i+1}:")
    print(f"  Patient Name: {event.patient_name}")
    print(f"  Device: {event.device}")
    print(f"  Event: {event.event}")
    print(f"  Event Time: {event.event_time}")
    print(f"  Time in Queue: {event.time_in_queue}")
    print(f"  Technician: {event.technician}")

print("\n" + "-" * 80)
print(f"\nTotal records found: {result.total_count}")

# Count events by type
event_types = {}
for event in result.data:
    event_types[event.event] = event_types.get(event.event, 0) + 1

print("\nEvent distribution:")
for event_type, count in sorted(event_types.items()):
    print(f"  {event_type}: {count}")
