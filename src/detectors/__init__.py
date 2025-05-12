"""
Detectors module - Implementation of various fault detectors
"""

from .fault_detector_factory import FaultDetectorFactory
from .spe_fault_detector import SPEFaultDetector
from .enhanced_transformer_detection import EnhancedTransformerModel

# Create alias for backward compatibility
create_fault_detector = FaultDetectorFactory.create_detector

__all__ = [
    'create_fault_detector',
    'FaultDetectorFactory',
    'SPEFaultDetector',
    'EnhancedTransformerModel',
] 