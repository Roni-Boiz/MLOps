from dataclasses import dataclass

@dataclass
class ModelNameConfig:
    """Model Configurations"""

    model_name: str = "RandomForest"
    fine_tuning: bool = True