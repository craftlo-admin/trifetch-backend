"""
ECG Signal Processing Module
Provides functions for detecting R-peaks, calculating RR intervals, and extracting heart rate features
"""
import numpy as np
from scipy import signal
from typing import List, Tuple, Dict


class ECGProcessor:
    """Process ECG signals to extract meaningful cardiac features"""
    
    def __init__(self, sampling_rate: int = 200):
        """
        Initialize ECG processor
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 200 Hz based on ECGData_200 files)
        """
        self.sampling_rate = sampling_rate
    
    def read_ecg_data(self, ecg_data: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse ECG data from comma-separated pairs
        
        Args:
            ecg_data: List of [value1, value2] string pairs
            
        Returns:
            Tuple of (lead1_signal, lead2_signal) as numpy arrays
        """
        lead1 = []
        lead2 = []
        
        for pair in ecg_data:
            lead1.append(float(pair[0]))
            lead2.append(float(pair[1]))
        
        return np.array(lead1), np.array(lead2)
    
    def detect_r_peaks(self, signal_data: np.ndarray, min_distance: int = 80) -> np.ndarray:
        """
        Detect R-peaks in ECG signal using adaptive thresholding
        
        Args:
            signal_data: 1D numpy array of ECG signal
            min_distance: Minimum samples between peaks (80 samples ~= 0.4s at 200Hz)
            
        Returns:
            Array of R-peak indices
        """
        # Apply bandpass filter (0.5-40 Hz) to remove baseline wander and high-frequency noise
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 40 / nyquist
        
        if len(signal_data) < 100:
            return np.array([])
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data)
        except:
            filtered_signal = signal_data
        
        # Square the signal to emphasize peaks
        squared = filtered_signal ** 2
        
        # Moving average to smooth
        window_size = int(0.12 * self.sampling_rate)  # 120ms window
        if window_size % 2 == 0:
            window_size += 1
        
        try:
            integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        except:
            integrated = squared
        
        # Find peaks with adaptive threshold
        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)
        
        return peaks
    
    def calculate_rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """
        Calculate RR intervals from R-peak indices
        
        Args:
            r_peaks: Array of R-peak indices
            
        Returns:
            Array of RR intervals in milliseconds
        """
        if len(r_peaks) < 2:
            return np.array([])
        
        # Calculate differences and convert to milliseconds
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
        
        return rr_intervals
    
    def calculate_heart_rate(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Calculate heart rate statistics from RR intervals
        
        Args:
            rr_intervals: Array of RR intervals in milliseconds
            
        Returns:
            Dictionary with heart rate statistics
        """
        if len(rr_intervals) == 0:
            return {
                'mean_hr': 0,
                'min_hr': 0,
                'max_hr': 0,
                'std_hr': 0
            }
        
        # Convert RR intervals (ms) to heart rate (bpm)
        heart_rates = 60000 / rr_intervals
        
        # Remove outliers (HR should be between 30-250 bpm)
        valid_hr = heart_rates[(heart_rates >= 30) & (heart_rates <= 250)]
        
        if len(valid_hr) == 0:
            valid_hr = heart_rates
        
        return {
            'mean_hr': float(np.mean(valid_hr)),
            'min_hr': float(np.min(valid_hr)),
            'max_hr': float(np.max(valid_hr)),
            'std_hr': float(np.std(valid_hr))
        }
    
    def calculate_hrv_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Calculate Heart Rate Variability (HRV) features
        
        Args:
            rr_intervals: Array of RR intervals in milliseconds
            
        Returns:
            Dictionary with HRV features
        """
        if len(rr_intervals) < 2:
            return {
                'rr_mean': 0,
                'rr_std': 0,
                'rr_min': 0,
                'rr_max': 0,
                'rr_range': 0,
                'rmssd': 0,  # Root mean square of successive differences
                'sdsd': 0,   # Standard deviation of successive differences
                'nn50': 0,   # Number of pairs of successive RR intervals > 50ms
                'pnn50': 0,  # Percentage of NN50
                'cv': 0      # Coefficient of variation
            }
        
        # Filter outliers (RR intervals should be 200-2000 ms for valid heart beats)
        valid_rr = rr_intervals[(rr_intervals >= 200) & (rr_intervals <= 2000)]
        
        if len(valid_rr) < 2:
            valid_rr = rr_intervals
        
        # Basic statistics
        rr_mean = float(np.mean(valid_rr))
        rr_std = float(np.std(valid_rr))
        
        # Successive differences
        successive_diffs = np.diff(valid_rr)
        
        # RMSSD
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
        
        # SDSD
        sdsd = float(np.std(successive_diffs))
        
        # NN50 and pNN50
        nn50 = int(np.sum(np.abs(successive_diffs) > 50))
        pnn50 = float(nn50 / len(successive_diffs) * 100) if len(successive_diffs) > 0 else 0
        
        # Coefficient of variation
        cv = float(rr_std / rr_mean) if rr_mean != 0 else 0
        
        return {
            'rr_mean': rr_mean,
            'rr_std': rr_std,
            'rr_min': float(np.min(valid_rr)),
            'rr_max': float(np.max(valid_rr)),
            'rr_range': float(np.max(valid_rr) - np.min(valid_rr)),
            'rmssd': rmssd,
            'sdsd': sdsd,
            'nn50': nn50,
            'pnn50': pnn50,
            'cv': cv
        }
    
    def extract_all_features(self, ecg_data: List[List[str]]) -> Dict[str, float]:
        """
        Extract all features from ECG data for ML classification
        
        Args:
            ecg_data: List of [value1, value2] string pairs
            
        Returns:
            Dictionary with all extracted features
        """
        # Parse ECG data
        lead1, lead2 = self.read_ecg_data(ecg_data)
        
        # Detect R-peaks in both leads
        r_peaks_lead1 = self.detect_r_peaks(lead1)
        r_peaks_lead2 = self.detect_r_peaks(lead2)
        
        # Calculate RR intervals
        rr_intervals_lead1 = self.calculate_rr_intervals(r_peaks_lead1)
        rr_intervals_lead2 = self.calculate_rr_intervals(r_peaks_lead2)
        
        # Use lead with more detected peaks
        if len(rr_intervals_lead1) >= len(rr_intervals_lead2):
            rr_intervals = rr_intervals_lead1
            r_peaks = r_peaks_lead1
            primary_signal = lead1
        else:
            rr_intervals = rr_intervals_lead2
            r_peaks = r_peaks_lead2
            primary_signal = lead2
        
        # Extract features
        hr_features = self.calculate_heart_rate(rr_intervals)
        hrv_features = self.calculate_hrv_features(rr_intervals)
        
        # Additional features
        features = {
            'num_beats': len(r_peaks),
            'signal_mean': float(np.mean(primary_signal)),
            'signal_std': float(np.std(primary_signal)),
            'signal_range': float(np.max(primary_signal) - np.min(primary_signal)),
        }
        
        # Combine all features
        features.update(hr_features)
        features.update(hrv_features)
        
        # Irregularity metrics (important for AFIB detection)
        if len(rr_intervals) > 3:
            # Calculate irregularity score
            rr_diffs = np.abs(np.diff(rr_intervals))
            features['irregularity_score'] = float(np.mean(rr_diffs))
            features['irregularity_max'] = float(np.max(rr_diffs))
        else:
            features['irregularity_score'] = 0
            features['irregularity_max'] = 0
        
        return features
