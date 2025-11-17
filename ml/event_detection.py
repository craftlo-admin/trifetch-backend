"""
Event Detection Module
Detects the exact position where cardiac arrhythmia starts in ECG signal
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from ml.ecg_processing import ECGProcessor


class EventDetector:
    """Detects exact timing of arrhythmia events in ECG signals"""
    
    def __init__(self, sampling_rate: int = 200):
        """
        Initialize event detector
        
        Args:
            sampling_rate: ECG sampling rate in Hz (200 Hz)
        """
        self.sampling_rate = sampling_rate
        self.processor = ECGProcessor(sampling_rate)
    
    def detect_afib_onset(self, rr_intervals: np.ndarray, r_peaks: np.ndarray) -> Optional[int]:
        """
        Detect AFIB onset by finding where irregularity starts
        
        AFIB is characterized by irregular RR intervals
        
        Args:
            rr_intervals: Array of RR intervals in milliseconds
            r_peaks: Array of R-peak indices
            
        Returns:
            Sample index where AFIB starts, or None
        """
        if len(rr_intervals) < 5:
            return None
        
        # Strategy 1: Check if AFIB present from start (early detection)
        # Look at first few beats to establish baseline
        early_window_size = min(5, len(rr_intervals))
        early_rr = rr_intervals[:early_window_size]
        
        # Calculate early irregularity
        if len(early_rr) >= 3:
            early_diffs = np.abs(np.diff(early_rr))
            early_irregularity = np.std(early_diffs)
            
            # Normal sinus rhythm: RR differences typically < 100ms
            # AFIB: RR differences often > 150ms
            if early_irregularity > 150:
                # AFIB likely present from start - return first R-peak
                return int(r_peaks[0])
        
        # Strategy 2: Look for sudden change from regular to irregular
        # Calculate rolling window irregularity
        window_size = 4
        irregularity_scores = []
        
        for i in range(len(rr_intervals) - window_size + 1):
            window = rr_intervals[i:i + window_size]
            diffs = np.abs(np.diff(window))
            irregularity = np.std(diffs)
            irregularity_scores.append(irregularity)
        
        if not irregularity_scores or len(irregularity_scores) < 3:
            return int(r_peaks[0])  # Default to start
        
        irregularity_scores = np.array(irregularity_scores)
        
        # Look for first sustained irregular period
        # Compare each window against a baseline (first few windows or absolute threshold)
        baseline_threshold = 150  # ms std deviation threshold
        
        # Find first window with high irregularity
        for i, score in enumerate(irregularity_scores):
            if score > baseline_threshold:
                # Return the R-peak position where irregularity starts
                if i < len(r_peaks):
                    return int(r_peaks[i])
        
        # If no clear irregular period, but we classified as AFIB
        # Return start of trace (AFIB throughout)
        return int(r_peaks[0])
    
    def detect_pause_onset(self, rr_intervals: np.ndarray, r_peaks: np.ndarray) -> Optional[int]:
        """
        Detect PAUSE onset by finding abnormally long RR interval
        
        PAUSE is characterized by RR interval > 2000ms (>2 seconds)
        
        Args:
            rr_intervals: Array of RR intervals in milliseconds
            r_peaks: Array of R-peak indices
            
        Returns:
            Sample index where PAUSE starts, or None
        """
        if len(rr_intervals) == 0:
            return None
        
        # Define pause threshold (2000ms = 2 seconds)
        pause_threshold = 2000  # milliseconds
        
        # Find first RR interval exceeding threshold
        pause_indices = np.where(rr_intervals > pause_threshold)[0]
        
        if len(pause_indices) > 0:
            # Return the R-peak position before the pause
            pause_index = pause_indices[0]
            if pause_index < len(r_peaks):
                return int(r_peaks[pause_index])
        
        # Alternative: Find RR interval that's significantly longer than average
        if len(rr_intervals) > 3:
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            
            # Look for RR interval > mean + 3*std
            outlier_indices = np.where(rr_intervals > mean_rr + 3 * std_rr)[0]
            
            if len(outlier_indices) > 0:
                outlier_index = outlier_indices[0]
                if outlier_index < len(r_peaks):
                    return int(r_peaks[outlier_index])
        
        return None
    
    def detect_vtach_onset(self, rr_intervals: np.ndarray, r_peaks: np.ndarray) -> Optional[int]:
        """
        Detect VTACH onset by finding where heart rate suddenly increases
        
        VTACH is characterized by heart rate > 100 bpm (RR < 600ms)
        
        Args:
            rr_intervals: Array of RR intervals in milliseconds
            r_peaks: Array of R-peak indices
            
        Returns:
            Sample index where VTACH starts, or None
        """
        if len(rr_intervals) < 3:
            return None
        
        # Convert RR intervals to heart rate
        heart_rates = 60000 / rr_intervals  # bpm
        
        # VTACH threshold: HR > 100 bpm (RR < 600ms)
        vtach_threshold_bpm = 100
        
        # Find where heart rate exceeds threshold for consecutive beats
        high_hr_indices = np.where(heart_rates > vtach_threshold_bpm)[0]
        
        if len(high_hr_indices) >= 3:
            # Look for 3+ consecutive high HR beats
            for i in range(len(high_hr_indices) - 2):
                if (high_hr_indices[i+1] == high_hr_indices[i] + 1 and 
                    high_hr_indices[i+2] == high_hr_indices[i] + 2):
                    # Found consecutive VTACH
                    onset_index = high_hr_indices[i]
                    if onset_index < len(r_peaks):
                        return int(r_peaks[onset_index])
        
        # Alternative: Find sudden HR increase
        if len(heart_rates) > 5:
            # Calculate rolling average heart rate
            window_size = 3
            for i in range(len(heart_rates) - window_size):
                baseline_hr = np.mean(heart_rates[i:i+window_size])
                next_hr = heart_rates[i+window_size]
                
                # If HR suddenly jumps by >30 bpm
                if next_hr - baseline_hr > 30:
                    onset_index = i + window_size
                    if onset_index < len(r_peaks):
                        return int(r_peaks[onset_index])
        
        return None
    
    def detect_event_start(self, ecg_data: List[List[str]], predicted_class: str) -> Dict:
        """
        Detect exact start time of arrhythmia event
        
        Args:
            ecg_data: List of [value1, value2] string pairs
            predicted_class: Predicted arrhythmia type (AFIB, PAUSE, VTACH)
            
        Returns:
            Dictionary with event detection results
        """
        # Parse ECG data
        lead1, lead2 = self.processor.read_ecg_data(ecg_data)
        
        # Detect R-peaks in both leads
        r_peaks_lead1 = self.processor.detect_r_peaks(lead1)
        r_peaks_lead2 = self.processor.detect_r_peaks(lead2)
        
        # Use lead with more detected peaks
        if len(r_peaks_lead1) >= len(r_peaks_lead2):
            r_peaks = r_peaks_lead1
            signal = lead1
        else:
            r_peaks = r_peaks_lead2
            signal = lead2
        
        # Calculate RR intervals
        rr_intervals = self.processor.calculate_rr_intervals(r_peaks)
        
        # Detect event onset based on predicted class
        onset_sample = None
        detection_method = ""
        
        if predicted_class == "AFIB":
            onset_sample = self.detect_afib_onset(rr_intervals, r_peaks)
            detection_method = "Irregular RR interval pattern detected"
        elif predicted_class == "PAUSE":
            onset_sample = self.detect_pause_onset(rr_intervals, r_peaks)
            detection_method = "Abnormally long RR interval detected"
        elif predicted_class == "VTACH":
            onset_sample = self.detect_vtach_onset(rr_intervals, r_peaks)
            detection_method = "Rapid heart rate increase detected"
        
        # Calculate time in seconds
        if onset_sample is not None:
            event_start_time = onset_sample / self.sampling_rate
            
            return {
                'detected': True,
                'event_start_second': round(event_start_time, 2),
                'event_start_sample': onset_sample,
                'detection_method': detection_method,
                'total_duration': len(ecg_data) / self.sampling_rate,
                'num_beats_analyzed': len(r_peaks)
            }
        else:
            # If no clear onset detected, assume it starts at beginning
            # or return middle of the trace
            fallback_time = len(ecg_data) / self.sampling_rate / 2
            
            return {
                'detected': False,
                'event_start_second': round(fallback_time, 2),
                'event_start_sample': len(ecg_data) // 2,
                'detection_method': 'No clear onset detected, returning midpoint',
                'total_duration': len(ecg_data) / self.sampling_rate,
                'num_beats_analyzed': len(r_peaks)
            }


# Global detector instance
_detector_instance = None


def get_event_detector() -> EventDetector:
    """
    Get or create global event detector instance (singleton pattern)
    
    Returns:
        EventDetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = EventDetector()
    
    return _detector_instance
