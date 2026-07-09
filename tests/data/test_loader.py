import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader

class TestDataLoader:
    """Test suite for DataLoader"""
    
    def test_loader_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        assert loader.data_dir == Path('data')
        assert loader.raw_data_dir == Path('data/raw')
        assert loader.processed_data_dir == Path('data/processed')
    
    def test_load_from_csv(self, tmp_path):
        """Test loading data from CSV"""
        # Create a test CSV
        test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        test_path = tmp_path / 'test.csv'
        test_df.to_csv(test_path, index=False)
        
        # Load and verify
        loader = DataLoader()
        loaded_df = loader.load_from_csv(test_path)
        
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ['col1', 'col2']
    
    def test_get_dataset_info(self):
        """Test dataset info extraction"""
        loader = DataLoader()
        test_df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        info = loader.get_dataset_info(test_df)
        assert info['shape'] == (4, 2)
        assert info['missing_values']['col1'] == 1
