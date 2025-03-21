import yaml
import os
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for DAPO."""
    
    def __init__(self, config_path: str):
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set environment variables if needed
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get(
            'CUDA_VISIBLE_DEVICES', self.get('system.device', 'cuda')
        )
        
    def get(self, key_path: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value using a dot-separated path.
        
        Args:
            key_path: Dot-separated path to the configuration value.
            default: Default value to return if path doesn't exist.
            
        Returns:
            The configuration value or default.
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def update(self, key_path: str, value: Any) -> None:
        """Update a configuration value using a dot-separated path.
        
        Args:
            key_path: Dot-separated path to the configuration value.
            value: New value to set.
        """
        keys = key_path.split('.')
        config = self.config
        
        for i, key in enumerate(keys[:-1]):
            if key not in config:
                config[key] = {}
            config = config[key]
                
        config[keys[-1]] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Returns:
            The configuration as a dictionary.
        """
        return self.config.copy()
