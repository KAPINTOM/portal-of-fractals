# Portal of Fractals

A sophisticated Python application for generating, visualizing, and saving high-quality fractal images using Tkinter and NumPy.

## Features

### Supported Fractals
- **Julia Set**: Dynamic visualization with various interesting regions and patterns
  - Dragon-like Pattern
  - Dendrite Pattern
  - Exotic Spiral
  - Spiraling Tendrils
  - Delicate Branches
  - Symmetric Pattern
  - Rabbit-like Set
  - Douady's Rabbit
  - Spiral Galaxy
  - Exotic Flower

- **Mandelbrot Set**: Exploration of classic regions
  - Spiral pattern
  - Mini Mandelbrot
  - Valley of spirals
  - Deep spiral
  - Detailed bulb
  - Spiral arms

- **Burning Ship Fractal**: Unique variation with distinctive regions
  - Main ship
  - Detailed hull
  - Side structures
  - Small ships
  - Bridge section
  - Antenna detail

### Technical Specifications

- **Resolution**:
  - Display: 1280x720
  - Export: 4K (3840x2160)

- **Performance Features**:
  - Multi-threaded rendering using ThreadPoolExecutor
  - Chunk-based processing for large images
  - Progress bar for real-time generation feedback
  - Caching system for recent fractals

### Color System
- Advanced coloring algorithm using:
  - Smooth iteration count
  - RGB multipliers (range 3-12)
  - Phase shifts (0-2π)
  - Contrast enhancement
  - Anti-aliasing via gaussian blur
  - Dark area protection

### User Interface
- **Main Window**: 1300x900 pixels
- **Components**:
  - Fractal type selection buttons
  - HD image save button
  - Progress bar
  - Canvas display (1280x720)
  - Information panel
  - Menu system

### Menu Options
1. **File Menu**
   - Save Current
   - Save Preferences
   - Exit

2. **View Menu**
   - Toggle Info Panel

### Information Display
- Mathematical formula
- Fractal classification
- Technical parameters
- Color settings
- Generation time
- Resolution details

### File Management
- Automatic creation of 'fractals' directory
- Timestamp-based file naming
- Configuration saving in JSON format
- Parameter export functionality

### Error Handling
- Comprehensive exception management
- Logging system
- Debug information
- Status messages

## Technical Implementation

### Core Components

1. **FractalGenerator Class**
   - Main application controller
   - UI management
   - Fractal generation coordination
   - Image processing and display

2. **Generation Methods**
   - `generate_julia()`
   - `generate_mandelbrot()`
   - `generate_burning_ship()`
   - `generate_fractal_chunk()`

3. **Image Processing**
   - `adjust_image_quality()`
   - `render_image()`
   - High-resolution export support

4. **UI Management**
   - Button state control
   - Progress updates
   - Information display
   - Event handling

### Dependencies
- tkinter
- PIL (Pillow)
- numpy
- scipy
- logging
- json
- concurrent.futures
- datetime
- random
- os

### Performance Optimizations
- Parallel processing for large images
- Efficient memory management
- Caching system
- Optimized numpy operations

## Image Quality Features

1. **Resolution Enhancement**
   - 4K export capability
   - LANCZOS resampling
   - Gaussian blur smoothing

2. **Color Enhancement**
   - Dynamic range adjustment
   - Contrast optimization
   - Dark area protection
   - Smooth gradient generation

## File Output

### Image Files
- Format: PNG
- Naming: `{fractal_type}_{timestamp}_4K.png`
- Location: `./fractals/` directory
- Optimization: PNG compression enabled

### Configuration Files
- Format: JSON
- Settings storage
- Parameter export
- Window preferences

## Logging System
- File: `fractal_generator.log`
- Level: INFO
- Format: Timestamp, Level, Message
- Exception tracking

## Error Management
- Graceful error handling
- User feedback
- Recovery mechanisms
- Debug information

## Project Structure
```
portal-of-fractals/
├── main.py
├── README.md
├── fractals/
├── fractal_config.json
└── fractal_generator.log
```