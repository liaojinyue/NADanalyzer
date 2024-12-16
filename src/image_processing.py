class ImageProcessor:
    def __init__(self, segmentation_method='otsu'):
        self.segmentation_method = segmentation_method
        
    def load_nd2(self, file_path):
        """Load and parse ND2 file"""
        pass
        
    def separate_channels(self, image):
        """Extract blue and green channels"""
        pass
    
    def segment_cells(self, image):
        """Apply selected segmentation method"""
        if self.segmentation_method == 'otsu':
            return self._otsu_segmentation(image)
        elif self.segmentation_method == 'watershed':
            return self._watershed_segmentation(image)
        # Add more methods as needed 