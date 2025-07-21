import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import random
from datetime import datetime
import time
import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    filename='fractal_generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FractalGenerator:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Portal of Fractals")
        self.window.geometry("1280x720")
        self.window.minsize(800, 600)
        
        # Variables for dynamic sizing
        self.canvas_width = 500
        self.canvas_height = 500
        self.current_image = None
        self.current_fractal_info = {}
        self.generation_thread = None
        self.stop_generation = threading.Event()
        self.info_window = None  # Reference to info window
        
        # Create button frames for better organization
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # Create label for fractal selection
        ttk.Label(button_frame, text="Fractal Type:").pack(side=tk.LEFT, padx=5)
        
        # Add initial value for C
        self.C = complex(-0.4, 0.6)  # Initial value for Julia set
        
        # Create buttons for different fractals
        self.julia_btn = ttk.Button(button_frame, text="Julia Set", 
                               command=lambda: self.start_fractal_generation('julia'))
        self.julia_btn.pack(side=tk.LEFT, padx=5)
        
        self.mandelbrot_btn = ttk.Button(button_frame, text="Mandelbrot Set", 
                                    command=lambda: self.start_fractal_generation('mandelbrot'))
        self.mandelbrot_btn.pack(side=tk.LEFT, padx=5)
        
        self.burning_ship_btn = ttk.Button(button_frame, text="Burning Ship", 
                                      command=lambda: self.start_fractal_generation('burning_ship'))
        self.burning_ship_btn.pack(side=tk.LEFT, padx=5)
        
        # Add Newton fractal button
        self.newton_btn = ttk.Button(button_frame, text="Newton Fractal", 
                                      command=lambda: self.start_fractal_generation('newton'))
        self.newton_btn.pack(side=tk.LEFT, padx=5)
        
        # Create save button
        self.save_btn = ttk.Button(button_frame, text="Save HD Image", command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=20)  # Added more padding to separate from fractal buttons

        # Create info button
        self.info_btn = ttk.Button(button_frame, text="Show Info", command=self.show_info_window)
        self.info_btn.pack(side=tk.LEFT, padx=10)

        # Create progress bar
        self.progress = ttk.Progressbar(self.window, length=300, mode='determinate')
        self.progress.pack(pady=5, fill=tk.X, padx=10)

        # Create canvas for displaying the fractal with dynamic sizing
        self.canvas_frame = ttk.Frame(self.window)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='#2b2b2b')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add initial text to the canvas
        self.initial_text_id = self.canvas.create_text(
            250, 250,  # Initial center position
            text="Select a fractal type to begin",
            fill="white",
            font=('Arial', 20),
            tags="initial_text"
        )
        
        # Bind to window resize event
        self.window.bind("<Configure>", self.on_window_resize)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.fractal_cache = {}  # Add cache dictionary
        self.max_cache_size = 5  # Keep last 5 fractals in memory
        
        self.create_menu()
        self.window.mainloop()
    
    def on_window_resize(self, event):
        """Handle window resize events"""
        # Only process if it's the main window being resized
        if event.widget == self.window:
            self.update_canvas_size()
    
    def on_canvas_resize(self, event):
        """Handle canvas resize events"""
        # Update canvas dimensions
        self.canvas_width = event.width
        self.canvas_height = event.height
        
        # Center the initial text
        self.canvas.coords(
            self.initial_text_id,
            self.canvas_width // 2,
            self.canvas_height // 2
        )
    
    def update_canvas_size(self):
        """Update canvas dimensions based on current size"""
        self.canvas_width = self.canvas.winfo_width()
        self.canvas_height = self.canvas.winfo_height()
        
        # Ensure minimum size
        if self.canvas_width < 100 or self.canvas_height < 100:
            self.canvas_width = 100
            self.canvas_height = 100

    def create_menu(self):
        """Create menu bar with additional options"""
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Current", command=self.save_image)
        file_menu.add_command(label="Save Preferences", command=self.save_preferences)
        file_menu.add_command(label="Export Parameters", command=self.export_parameters)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.window.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Information", command=self.show_info_window)

    def show_info_window(self):
        """Show the fractal information in a resizable popup window"""
        if self.info_window and self.info_window.winfo_exists():
            # Bring existing window to front
            self.info_window.lift()
            self.info_window.deiconify()
            return
        
        # Create new info window
        self.info_window = tk.Toplevel(self.window)
        self.info_window.title("Fractal Information")
        self.info_window.geometry("800x600")
        self.info_window.minsize(400, 300)
        
        # Create frame for text and scrollbar
        frame = ttk.Frame(self.info_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create text area for fractal information
        self.info_text = tk.Text(
            frame, 
            wrap=tk.WORD, 
            yscrollcommand=scrollbar.set,
            font=('Courier New', 10)  # Monospaced for better formatting
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.info_text.yview)
        
        # Add current info if available
        if self.current_fractal_info:
            self.update_info()
        
        # Make window resizable
        self.info_window.resizable(True, True)

    def update_progress(self, value):
        """Thread-safe progress update"""
        if hasattr(self, 'window'):
            try:
                self.window.after(0, self._update_progress_safe, value)
            except tk.TclError:
                pass

    def _update_progress_safe(self, value):
        """Internal method for actual progress update"""
        if hasattr(self, 'progress'):
            try:
                self.progress['value'] = value
                self.window.update_idletasks()
            except tk.TclError:
                pass

    def update_info(self):
        if not self.current_fractal_info:
            return
            
        info = self.current_fractal_info
        
        try:
            fractal_type = info.get('type', 'Unknown')
            parameters = info.get('parameters', '')
            color_info = info.get('color_info', 'N/A')
            phase_info = info.get('phase_info', 'N/A')
            
            # Extract parameters based on fractal type
            if fractal_type == 'Julia Set':
                # Parse the complex C value from the parameters string
                c_part = parameters.split(',')[0]  # Get first part containing C value
                c_str = c_part.split(': ')[1]  # Get the actual complex number string
                c_value = complex(c_str)
                
                math_formula = f"""Mathematical Formula:
zₙ₊₁ = zₙ² + c
where c = {c_value}
Escape Criterion: |z| > 2
Domain: Complex plane bounded by [-1.5, 1.5] × [-1, 1]"""

                # Julia set classification
                julia_types = {
                    complex(-0.4, 0.6): "Dragon-like Pattern",
                    complex(0.285, 0): "Dendrite Pattern",
                    complex(0.45, 0.1428): "Exotic Spiral",
                    complex(-0.70176, -0.3842): "Spiraling Tendrils",
                    complex(-0.835, -0.2321): "Delicate Branches",
                    complex(0.35, 0.35): "Symmetric Pattern",
                    complex(0, 0.8): "Rabbit-like Set",
                    complex(-0.123, 0.745): "Douady's Rabbit",
                    complex(-0.391, -0.587): "Spiral Galaxy",
                    complex(-0.54, 0.54): "Exotic Flower"
                }
                
                pattern_type = "Unknown Pattern"
                for c_key, pattern in julia_types.items():
                    if abs(c_value - c_key) < 0.0001:
                        pattern_type = pattern
                        break

                classification = f"""Julia Set Classification:
Pattern Type: {pattern_type}
Connected Set: {abs(c_value) <= 2}
Symmetry: {"Real Axis" if abs(c_value.imag) < 0.0001 else "Complex Plane"}"""

            elif fractal_type == 'Mandelbrot Set':
                math_formula = """Mathematical Formula:
zₙ₊₁ = zₙ² + c
where z₀ = 0
Escape Criterion: |z| > 2
Domain: Complex plane c"""

                classification = """Mandelbrot Set Classification:
A point c is in the Mandelbrot set if the sequence remains bounded
Main cardioid: Points where |c| ≤ 1/4
Period-2 bulb: Points where c ≈ -1"""

            elif fractal_type == 'Burning Ship Fractal':
                math_formula = """Mathematical Formula:
zₙ₊₁ = (|Re(zₙ)| + |Im(zₙ)|i)² + c
where z₀ = 0
Escape Criterion: |z| > 2
Domain: Complex plane c"""

                classification = """Burning Ship Classification:
Variation of the Mandelbrot set using absolute values
Creates flame-like structures and a distinctive 'ship' shape"""
            
            elif fractal_type == 'Newton Fractal':
                poly_info = info.get('polynomial', 'Unknown polynomial')
                roots_info = info.get('roots', 'Unknown roots')
                
                math_formula = f"""Mathematical Formula:
Newton-Raphson Iteration:
zₙ₊₁ = zₙ - p(zₙ)/p'(zₙ)
where p(z) = {poly_info}
Roots: {roots_info}
Convergence Criterion: |p(z)| < 1e-5
Domain: Complex plane centered at (0,0) with scale ±1.5"""

                classification = """Newton Fractal Classification:
Visualizes basins of attraction for roots of polynomial
Colors represent which root a starting point converges to
Brightness represents speed of convergence"""

            # Color algorithm description
            color_info = f"""Color Algorithm:
RGB = (sin(s·r_m/30 + φr)·127 + 128, sin(s·g_m/30 + φg)·127 + 128, sin(s·b_m/30 + φb)·127 + 128)
where s = iteration + 1 - log₂(log₂(|z|)) for escape fractals
For Newton: s = iteration count
{color_info}
{phase_info}"""

            # Technical parameters
            tech_info = f"""Technical Parameters:
Resolution: {self.canvas_width}x{self.canvas_height}
Maximum Iterations: {info.get('iterations', 'N/A')}
Generation Time: {info.get('time', 'N/A')} seconds
{parameters}"""

            # Combine all information
            full_info = f"""=== {fractal_type} Technical Information ===

{math_formula}

{classification}

{color_info}

{tech_info}"""
            
            # Update text widget only if info window exists
            if self.info_window and self.info_text:
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, full_info)
            
        except Exception as e:
            error_info = f"Error displaying information: {str(e)}"
            print(f"Debug - Error details: {e}")  # For debugging
            if self.info_window and self.info_text:
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, error_info)

    def generate_fractal_chunk(self, params):
        """Generate a portion of the fractal"""
        start_row, end_row, width, height, C, color_mults, color_phases, x_range, y_range, max_iter = params
        chunk = np.zeros((end_row - start_row, width, 3), dtype=np.uint8)
        
        # Use the provided x and y ranges
        x = x_range
        y = y_range[start_row:end_row]
        
        # Create mesh grid for this chunk
        X, Y = np.meshgrid(x, y)
        Z = X + Y*1j
        
        # Use the received color parameters
        r_mult, g_mult, b_mult = color_mults
        phase_r, phase_g, phase_b = color_phases
        
        iteration_count = np.zeros_like(Z, dtype=int)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C
            iteration_count[mask] = i

        # Create smooth_iter array
        smooth_iter = np.full_like(iteration_count, max_iter, dtype=float)   # inside points set to max_iter

        # For points that escaped (|Z|>2) we compute the fractional iteration
        escaped = np.abs(Z) > 2
        if np.any(escaped):
            # Only compute for escaped points
            log_zn = np.log2(np.abs(Z[escaped]))
            # Add small epsilon to avoid log(0)
            log_zn = np.log2(np.abs(Z[escaped]) + 1e-10)
            smooth_iter[escaped] = iteration_count[escaped] + 1 - np.log2(log_zn)

        # Apply coloring
        chunk[:,:,0] = np.sin(smooth_iter * r_mult/30 + phase_r) * 127 + 128
        chunk[:,:,1] = np.sin(smooth_iter * g_mult/30 + phase_g) * 127 + 128
        chunk[:,:,2] = np.sin(smooth_iter * b_mult/30 + phase_b) * 127 + 128

        # Ensure no pure black areas
        dark_mask = np.all(chunk < 30, axis=2)
        chunk[dark_mask] = [30, 30, 30]
        
        # Ensure valid values
        chunk = np.nan_to_num(chunk, nan=30)
        chunk = np.clip(chunk, 30, 255).astype(np.uint8)

        return chunk

    def generate_julia(self):
        start_time = time.time()
        max_iter = 1000
        
        # Update canvas dimensions
        self.update_canvas_size()
        width = self.canvas_width
        height = self.canvas_height

        # Interesting Julia set parameters and positions
        julia_regions = [
            {
                'C': complex(-0.4, 0.6), 
                'center': (0.0, 0.0),  # Centered view
                'zoom': 1.0
            },  # Full view
            {
                'C': complex(0.285, 0), 
                'center': (0.0, 0.0),
                'zoom': 1.0
            },  # Full view
            {
                'C': complex(0.45, 0.1428),
                'center': (0.0, 0.0),
                'zoom': 1.0
            },  # Full view
            {
                'C': complex(-0.70176, -0.3842),
                'center': (-0.2, 0.1),
                'zoom': 1.5
            },  # Slightly zoomed
            {
                'C': complex(-0.835, -0.2321),
                'center': (0.1, -0.3),
                'zoom': 2.0
            },  # Zoomed view
            {
                'C': complex(0.35, 0.35),
                'center': (-0.15, 0.15),
                'zoom': 1.5
            }  # Zoomed view
        ]
        
        # Random selection of region and parameters
        region = random.choice(julia_regions)
        self.C = region['C']
        base_zoom = region['zoom']
        zoom = base_zoom * random.uniform(0.8, 1.2)
        center_x, center_y = region['center']
        
        # Add some random variation to the center position
        center_x += random.uniform(-0.1, 0.1)
        center_y += random.uniform(-0.1, 0.1)
        
        # Continue with generation code
        x_range_val = 3.0 / zoom
        y_range_val = x_range_val * height / width
        x = np.linspace(center_x - x_range_val/2, center_x + x_range_val/2, width)
        y = np.linspace(center_y - y_range_val/2, center_y + y_range_val/2, height)
        
        # Split into chunks for parallel processing
        chunks = 4
        chunk_size = height // chunks
        self.update_progress(0)
        
        # Generate color parameters once for all chunks
        r_mult = random.randint(3, 12)
        g_mult = random.randint(3, 12)
        b_mult = random.randint(3, 12)
        phase_r = random.random() * 2 * np.pi
        phase_g = random.random() * 2 * np.pi
        phase_b = random.random() * 2 * np.pi
        
        # Initialize array for final image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(chunks):
                start_row = i * chunk_size
                end_row = start_row + chunk_size if i < chunks - 1 else height
                
                params = (
                    start_row, 
                    end_row, 
                    width, 
                    height, 
                    self.C,
                    (r_mult, g_mult, b_mult), 
                    (phase_r, phase_g, phase_b),
                    x,  # Add the calculated x and y ranges
                    y,
                    max_iter  # Pass max_iter to the chunk function
                )
                futures.append(executor.submit(self.generate_fractal_chunk, params))
                self.update_progress((i + 1) * 25)
        
            # Combine all chunks
            image = np.vstack([f.result() for f in futures])
    
        self.update_progress(100)
        
        # Add fractal information with enhanced parameters
        self.current_fractal_info = {
            'type': 'Julia Set',
            'parameters': (
                f'C: {self.C}, '
                f'Center: ({center_x:.4f}, {center_y:.4f}), '
                f'Zoom: {zoom:.1f}x'
            ),
            'iterations': max_iter,
            'color_info': f'RGB multipliers: ({r_mult}, {g_mult}, {b_mult})',
            'phase_info': f'Phase shifts: ({phase_r:.4f}, {phase_g:.4f}, {phase_b:.4f})',
            'time': f"{time.time() - start_time:.2f}"
        }

        return image

    def generate_mandelbrot(self):
        start_time = time.time()
        
        # Update canvas dimensions
        self.update_canvas_size()
        width = self.canvas_width
        height = self.canvas_height
        
        max_iter = 1000
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Interesting regions in the Mandelbrot set
        regions = [
            {'center': (-0.7435, 0.1314), 'zoom': 150},  # Spiral pattern
            {'center': (-0.16070135, 1.0375665), 'zoom': 80},  # Mini Mandelbrot
            {'center': (-0.5225, 0.6234), 'zoom': 125},  # Valley of spirals
            {'center': (-0.7443, 0.1315), 'zoom': 200},  # Deep spiral
            {'center': (-0.15652, 1.03225), 'zoom': 250},  # Detailed bulb
            {'center': (-0.722, 0.246), 'zoom': 100},  # Spiral arms
        ]
        
        # Random selection of region and zoom variation
        region = random.choice(regions)
        center_x, center_y = region['center']
        zoom = region['zoom'] * random.uniform(0.5, 2.0)
        
        # Calculate boundaries based on zoom and center
        x_range = 3.0 / zoom
        y_range = x_range * height / width
        x = np.linspace(center_x - x_range/2, center_x + x_range/2, width)
        y = np.linspace(center_y - y_range/2, center_y + y_range/2, height)
        X, Y = np.meshgrid(x, y)
        C = X + Y*1j
        Z = np.zeros_like(C)
        
        # Enhanced color parameters - Modified this part
        r_mult = random.randint(5, 15)  # Increased range
        g_mult = random.randint(5, 15)
        b_mult = random.randint(5, 15)
        
        # Phase shifts for more vibrant colors
        phase_r = random.uniform(0, 4 * np.pi)  # Increased phase range
        phase_g = random.uniform(0, 4 * np.pi)
        phase_b = random.uniform(0, 4 * np.pi)
        
        # Color offset for more brightness
        offset_r = random.uniform(0.5, 1.0)
        offset_g = random.uniform(0.5, 1.0)
        offset_b = random.uniform(0.5, 1.0)
        
        iteration_count = np.zeros_like(Z, dtype=int)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            iteration_count[mask] = i
            
            # Update progress continuously
            progress_value = min(100, (i * 100) // max_iter)
            self.update_progress(progress_value)
        
        # Create smooth_iter array
        smooth_iter = np.full_like(iteration_count, max_iter, dtype=float)   # inside points set to max_iter

        # For points that escaped (|Z|>2) we compute the fractional iteration
        escaped = np.abs(Z) > 2
        if np.any(escaped):
            # Only compute for escaped points
            log_zn = np.log2(np.abs(Z[escaped]) + 1e-10)
            smooth_iter[escaped] = iteration_count[escaped] + 1 - np.log2(log_zn)
        
        # Ensure valid values
        smooth_iter = np.nan_to_num(smooth_iter, nan=max_iter)
        
        # Modified color application for more brightness
        image[:,:,0] = np.sin(smooth_iter * r_mult/30 + phase_r) * 127 * offset_r + 128
        image[:,:,1] = np.sin(smooth_iter * g_mult/30 + phase_g) * 127 * offset_g + 128
        image[:,:,2] = np.sin(smooth_iter * b_mult/30 + phase_b) * 127 * offset_b + 128
        
        # Ensure brighter colors
        image = np.clip(image * 1.2, 30, 255).astype(np.uint8)
        
        # Avoid very dark areas
        dark_mask = np.all(image < 50, axis=2)
        image[dark_mask] = [50, 50, 50]
        
        # Add saturation variation
        gray = np.mean(image, axis=2, keepdims=True)
        saturation = random.uniform(0.8, 1.2)  # Random saturation factor
        image = np.clip((image - gray) * saturation + gray, 30, 255).astype(np.uint8)
        
        self.current_fractal_info = {
            'type': 'Mandelbrot Set',
            'parameters': f'Center: ({center_x:.4f}, {center_y:.4f}), Zoom: {zoom:.1f}x',
            'iterations': max_iter,
            'color_info': f'RGB multipliers: ({r_mult}, {g_mult}, {b_mult})',
            'phase_info': f'Phase shifts: ({phase_r:.4f}, {phase_g:.4f}, {phase_b:.4f})',
            'time': f"{time.time() - start_time:.2f}"
        }
        
        return image

    def generate_burning_ship(self):
        start_time = time.time()
        
        # Update canvas dimensions
        self.update_canvas_size()
        width = self.canvas_width
        height = self.canvas_height
        
        max_iter = 1000
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Interesting regions in the Burning Ship fractal
        regions = [
            {'center': (-1.8, -0.05), 'zoom': 1.0},  # Full view
            {'center': (-1.75, -0.04), 'zoom': 1.5},  # Zoomed view
            {'center': (-1.7, -0.03), 'zoom': 2.0},  # More zoomed
            {'center': (-1.65, -0.02), 'zoom': 2.5},  # Detailed view
            {'center': (-1.8, -0.08), 'zoom': 1.2},  # Bridge section
            {'center': (-1.73, -0.01), 'zoom': 1.8},  # Antenna detail,
        ]
        
        # Random selection of region and zoom variation
        region = random.choice(regions)
        center_x, center_y = region['center']
        zoom = region['zoom'] * random.uniform(0.8, 1.2)
        
        # Add some random variation to center
        center_x += random.uniform(-0.05, 0.05)
        center_y += random.uniform(-0.05, 0.05)
        
        # Calculate boundaries based on zoom and center
        x_range = 3.0 / zoom
        y_range = x_range * height / width
        x = np.linspace(center_x - x_range/2, center_x + x_range/2, width)
        y = np.linspace(center_y - y_range/2, center_y + y_range/2, height)
        X, Y = np.meshgrid(x, y)
        C = X + Y*1j
        Z = np.zeros_like(C)
        
        # Enhanced color parameters
        r_mult = random.randint(5, 15)
        g_mult = random.randint(5, 15)
        b_mult = random.randint(5, 15)
        
        # Phase shifts for vibrant colors
        phase_r = random.uniform(0, 4 * np.pi)
        phase_g = random.uniform(0, 4 * np.pi)
        phase_b = random.uniform(0, 4 * np.pi)
        
        # Color offsets
        offset_r = random.uniform(0.5, 1.0)
        offset_g = random.uniform(0.5, 1.0)
        offset_b = random.uniform(0.5, 1.0)
        
        iteration_count = np.zeros_like(Z, dtype=int)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            # Burning Ship formula: (|Re| + |Im|i)^2 + c
            real_part = np.abs(Z[mask].real)
            imag_part = np.abs(Z[mask].imag)
            Z[mask] = (real_part + imag_part*1j)**2 + C[mask]
            iteration_count[mask] = i
            
            # Update progress continuously
            progress_value = min(100, (i * 100) // max_iter)
            self.update_progress(progress_value)
        
        # Create smooth_iter array
        smooth_iter = np.full_like(iteration_count, max_iter, dtype=float)
        
        # For escaped points
        escaped = np.abs(Z) > 2
        if np.any(escaped):
            log_zn = np.log2(np.abs(Z[escaped]) + 1e-10)
            smooth_iter[escaped] = iteration_count[escaped] + 1 - np.log2(log_zn)
        
        # Ensure valid values
        smooth_iter = np.nan_to_num(smooth_iter, nan=max_iter)
        
        # Apply coloring with offsets
        image[:,:,0] = np.sin(smooth_iter * r_mult/30 + phase_r) * 127 * offset_r + 128
        image[:,:,1] = np.sin(smooth_iter * g_mult/30 + phase_g) * 127 * offset_g + 128
        image[:,:,2] = np.sin(smooth_iter * b_mult/30 + phase_b) * 127 * offset_b + 128
        
        # Brighten image
        image = np.clip(image * 1.2, 30, 255).astype(np.uint8)
        
        # Avoid dark areas
        dark_mask = np.all(image < 50, axis=2)
        image[dark_mask] = [50, 50, 50]
        
        # Add saturation variation
        gray = np.mean(image, axis=2, keepdims=True)
        saturation = random.uniform(0.8, 1.2)
        image = np.clip((image - gray) * saturation + gray, 30, 255).astype(np.uint8)
        
        self.current_fractal_info = {
            'type': 'Burning Ship Fractal',
            'parameters': f'Center: ({center_x:.4f}, {center_y:.4f}), Zoom: {zoom:.1f}x',
            'iterations': max_iter,
            'color_info': f'RGB multipliers: ({r_mult}, {g_mult}, {b_mult})',
            'phase_info': f'Phase shifts: ({phase_r:.4f}, {phase_g:.4f}, {phase_b:.4f})',
            'time': f"{time.time() - start_time:.2f}"
        }
        
        return image

    def generate_newton_chunk(self, params):
        """Generate a portion of the Newton fractal"""
        start_row, end_row, width, height, roots, color_mults, color_phases, x_range, y_range, max_iter = params
        
        # Extract color parameters
        r_mult, g_mult, b_mult = color_mults
        phase_r, phase_g, phase_b = color_phases
        
        # Create the grid for this chunk
        x = x_range
        y = y_range[start_row:end_row]
        X, Y = np.meshgrid(x, y)
        Z = X + Y * 1j
        
        # Define polynomial and derivative for the given roots
        def p(z):
            return (z - roots[0]) * (z - roots[1]) * (z - roots[2])
        
        def p_prime(z):
            term1 = (z - roots[1]) * (z - roots[2])
            term2 = (z - roots[0]) * (z - roots[2])
            term3 = (z - roots[0]) * (z - roots[1])
            return term1 + term2 + term3
        
        # Create arrays for iteration count and root indices
        iteration_count = np.zeros(Z.shape, dtype=int)
        root_index = np.zeros(Z.shape, dtype=int) - 1  # -1 means not converged
        
        # Mask for points that haven't converged
        mask = np.ones(Z.shape, dtype=bool)
        
        # Newton-Raphson iteration
        for i in range(max_iter):
            Z_active = Z[mask]
            f_val = p(Z_active)
            f_prime_val = p_prime(Z_active)
            
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                step = f_val / (f_prime_val + 1e-20)
            
            Z[mask] = Z_active - step
            f_val_new = p(Z[mask])
            
            # Check convergence (|f(z)| < tolerance)
            converged = np.abs(f_val_new) < 1e-5
            iteration_count[mask] = np.where(converged, i, iteration_count[mask])
            
            # For converged points, find the closest root
            if np.any(converged):
                # Compute distances to roots for converged points
                dist = np.abs(Z_active[converged, None] - np.array(roots)[None, :])
                closest = np.argmin(dist, axis=1)
                
                # Fix: Create a temporary array to avoid broadcasting issues
                active_root = root_index[mask]
                active_root[converged] = closest
                root_index[mask] = active_root
            
            # Update mask: remove converged points
            mask[mask] = ~converged
            
            # If no points left, break
            if not np.any(mask):
                break
        
        # For any remaining points that didn't converge, assign to the closest root
        unconverged = (root_index == -1)
        if np.any(unconverged):
            dist = np.abs(Z[unconverged, None] - np.array(roots)[None, :])
            closest = np.argmin(dist, axis=1)
            root_index[unconverged] = closest
            iteration_count[unconverged] = max_iter
        
        # Convert root index to a phase shift (for coloring)
        num_roots = len(roots)
        root_phase = root_index * (2 * np.pi / num_roots)
        
        # Create color chunk
        chunk = np.zeros((end_row - start_row, width, 3), dtype=np.uint8)
        smooth_iter = iteration_count.astype(float)
        
        # Apply coloring with root phase shift
        # FIX: Ensure we use the correct dimensions for assignment
        r_channel = np.sin(smooth_iter * r_mult/30 + phase_r + root_phase) * 127 + 128
        g_channel = np.sin(smooth_iter * g_mult/30 + phase_g + root_phase) * 127 + 128
        b_channel = np.sin(smooth_iter * b_mult/30 + phase_b + root_phase) * 127 + 128
        
        # Assign to chunk
        chunk[:, :, 0] = r_channel
        chunk[:, :, 1] = g_channel
        chunk[:, :, 2] = b_channel
        
        # Ensure no pure black
        dark_mask = np.all(chunk < 30, axis=2)
        chunk[dark_mask] = [30, 30, 30]
        
        return chunk

    def generate_newton(self):
        start_time = time.time()
        
        # Update canvas dimensions
        self.update_canvas_size()
        width = self.canvas_width
        height = self.canvas_height
        
        max_iter = 50
        
        # Define interesting regions for Newton fractal
        regions = [
            {'center': (0.0, 0.0), 'zoom': 1.0},    # Full view
            {'center': (-0.5, -0.5), 'zoom': 1.5},  # Top-left quadrant
            {'center': (0.5, -0.5), 'zoom': 1.5},   # Top-right quadrant
            {'center': (-0.5, 0.5), 'zoom': 1.5},   # Bottom-left quadrant
            {'center': (0.5, 0.5), 'zoom': 1.5},    # Bottom-right quadrant
            {'center': (0.0, 0.0), 'zoom': 2.0},    # Zoomed center
            {'center': (1.0, 0.0), 'zoom': 2.0},    # Zoomed right
            {'center': (-1.0, 0.0), 'zoom': 2.0},   # Zoomed left
            {'center': (0.0, 1.0), 'zoom': 2.0},    # Zoomed top
            {'center': (0.0, -1.0), 'zoom': 2.0},   # Zoomed bottom
        ]
        
        # Random selection of region
        region = random.choice(regions)
        center_x, center_y = region['center']
        zoom = region['zoom'] * random.uniform(0.8, 1.2)
        
        # Calculate boundaries based on zoom and center
        x_range_val = 3.0 / zoom
        y_range_val = x_range_val * height / width
        x = np.linspace(center_x - x_range_val/2, center_x + x_range_val/2, width)
        y = np.linspace(center_y - y_range_val/2, center_y + y_range_val/2, height)
        
        # Create random polynomial with complex roots
        roots = []
        for _ in range(3):  # Cubic polynomial
            real = random.uniform(-1.5, 1.5)
            imag = random.uniform(-1.5, 1.5)
            roots.append(complex(real, imag))
        
        # Format polynomial for display
        poly_str = f"(z - ({roots[0].real:.2f}+{roots[0].imag:.2f}i))"
        poly_str += f"(z - ({roots[1].real:.2f}+{roots[1].imag:.2f}i))"
        poly_str += f"(z - ({roots[2].real:.2f}+{roots[2].imag:.2f}i))"
        
        # Format roots for display
        roots_str = ", ".join([f"{r.real:.2f}{r.imag:+.2f}i" for r in roots])
        
        # Enhanced color parameters
        r_mult = random.randint(5, 15)
        g_mult = random.randint(5, 15)
        b_mult = random.randint(5, 15)
        phase_r = random.uniform(0, 4 * np.pi)
        phase_g = random.uniform(0, 4 * np.pi)
        phase_b = random.uniform(0, 4 * np.pi)
        
        # Split into chunks for parallel processing
        chunks = 4
        chunk_size = height // chunks
        self.update_progress(0)
        
        # Initialize array for final image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(chunks):
                start_row = i * chunk_size
                end_row = start_row + chunk_size if i < chunks - 1 else height
                
                params = (
                    start_row, 
                    end_row, 
                    width, 
                    height, 
                    roots, 
                    (r_mult, g_mult, b_mult), 
                    (phase_r, phase_g, phase_b),
                    x,  # x_range
                    y,  # y_range
                    max_iter
                )
                futures.append(executor.submit(self.generate_newton_chunk, params))
            
            # Combine chunks as they complete
            for i, future in enumerate(futures):
                chunk = future.result()
                start_row = i * chunk_size
                end_row = start_row + chunk_size if i < chunks - 1 else height
                image[start_row:end_row, :, :] = chunk
                self.update_progress((i + 1) * (100 / chunks))
        
        self.update_progress(100)
        
        # Brighten image
        image = np.clip(image * 1.1, 30, 255).astype(np.uint8)
        
        # Avoid dark areas
        dark_mask = np.all(image < 50, axis=2)
        image[dark_mask] = [50, 50, 50]
        
        # Store fractal information
        self.current_fractal_info = {
            'type': 'Newton Fractal',
            'parameters': f'Center: ({center_x:.4f}, {center_y:.4f}), Zoom: {zoom:.1f}x',
            'iterations': max_iter,
            'color_info': f'RGB multipliers: ({r_mult}, {g_mult}, {b_mult})',
            'phase_info': f'Phase shifts: ({phase_r:.4f}, {phase_g:.4f}, {phase_b:.4f})',
            'polynomial': poly_str,
            'roots': roots_str,
            'time': f"{time.time() - start_time:.2f}"
        }
        
        return image

    def start_fractal_generation(self, fractal_type):
        """Start fractal generation in a separate thread"""
        # Cancel any ongoing generation
        if self.generation_thread and self.generation_thread.is_alive():
            self.stop_generation.set()
            self.generation_thread.join(timeout=0.5)
        
        # Reset stop flag
        self.stop_generation.clear()
        
        # Disable buttons during generation
        self.disable_buttons()
        self.progress['value'] = 0
        self.window.update_idletasks()
        
        # Remove initial text if it exists
        if self.initial_text_id:
            self.canvas.delete(self.initial_text_id)
            self.initial_text_id = None
        
        # Start new thread for generation
        self.generation_thread = threading.Thread(
            target=self.generate_fractal,
            args=(fractal_type,),
            daemon=True
        )
        self.generation_thread.start()

    def generate_fractal(self, fractal_type='julia'):
        """Generate fractal in a background thread"""
        try:
            self.update_progress(0)
            
            # Generate the fractal
            if fractal_type == 'julia':
                image_array = self.generate_julia()
            elif fractal_type == 'mandelbrot':
                image_array = self.generate_mandelbrot()
            elif fractal_type == 'burning_ship':
                image_array = self.generate_burning_ship()
            elif fractal_type == 'newton':
                image_array = self.generate_newton()
            else:
                image_array = self.generate_julia()  # Default to Julia
                
            self.update_progress(75)
            
            # Apply quality improvements
            image_array = self.adjust_image_quality(image_array)
            
            # Render the image
            if self.render_image(image_array):
                self.update_info()
                self.update_progress(100)  # Ensure completion
            else:
                raise Exception("Failed to render image")
                
        except Exception as e:
            error_msg = f"Error generating fractal: {str(e)}"
            print(error_msg)
            logging.error(error_msg, exc_info=True)
            # Display error in info text
            if self.info_window and self.info_text:
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, error_msg)
        finally:
            self.enable_buttons()

    def disable_buttons(self):
        """Disable all buttons"""
        for button in [self.julia_btn, self.mandelbrot_btn, 
                      self.burning_ship_btn, self.newton_btn, 
                      self.save_btn, self.info_btn]:
            button['state'] = 'disabled'

    def enable_buttons(self):
        """Enable all buttons"""
        for button in [self.julia_btn, self.mandelbrot_btn, 
                      self.burning_ship_btn, self.newton_btn, 
                      self.save_btn, self.info_btn]:
            button['state'] = 'normal'

    def save_image(self):
        """Save the last generated fractal in high resolution"""
        if not self.current_fractal_info or not self.current_image:
            print("No fractal generated to save")
            return
            
        self.save_btn['state'] = 'disabled'
        try:
            # Create directory if it doesn't exist
            if not os.path.exists('fractals'):
                os.makedirs('fractals')
            
            # Get information about the current fractal
            fractal_type = self.current_fractal_info['type']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate file name
            filename = f"fractals/{fractal_type.replace(' ', '_')}_{timestamp}_4K.png"
            
            # Resize to 4K
            image_4k = self.current_image.resize((3840, 2160), Image.Resampling.LANCZOS)
            
            # Save image
            image_4k.save(filename, "PNG", optimize=True)
            
            print(f"Image saved successfully as {filename}")
            self.update_progress(100)
            
        except Exception as e:
            print(f"Error saving image: {e}")
            logging.error(f"Error saving image: {e}", exc_info=True)
        finally:
            self.save_btn['state'] = 'normal'

    def save_preferences(self):
        """Save current settings to a config file"""
        config = {
            'window_size': self.window.geometry(),
            'last_fractal_type': self.current_fractal_info.get('type'),
            'max_iterations': 1000,
            'color_scheme': {
                'r_mult_range': (3, 12),
                'g_mult_range': (3, 12),
                'b_mult_range': (3, 12)
            }
        }
        with open('fractal_config.json', 'w') as f:
            json.dump(config, f)

    def export_parameters(self):
        """Export current fractal parameters for reproduction"""
        if not self.current_fractal_info:
            return
            
        params = {
            'fractal_type': self.current_fractal_info['type'],
            'parameters': self.current_fractal_info['parameters'],
            'color_settings': {
                'multipliers': self.current_fractal_info['color_info'],
                'phases': self.current_fractal_info['phase_info']
            }
        }
        filename = f"fractal_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)

    def adjust_image_quality(self, image):
        """Enhance image quality with post-processing"""
        # Enhance contrast
        p2, p98 = np.percentile(image, (2, 98))
        image = np.clip(image, p2, p98)
        image = ((image - p2) / (p98 - p2) * 255).astype(np.uint8)
        
        return image

    def render_image(self, image_array):
        """Centralized method to render images"""
        # Check for empty image
        if image_array is None or image_array.size == 0:
            logging.error("Empty image array in render_image")
            return False
            
        try:
            # Ensure the image is in the correct format
            image_array = image_array.astype(np.uint8)
            
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_array)
            
            # Save reference to the current image
            self.current_image = image
            
            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(image)
            self.canvas.delete("all")  # Clear canvas before displaying new image
            
            # Create image at center of current canvas
            self.canvas.create_image(
                self.canvas_width//2, 
                self.canvas_height//2, 
                image=photo
            )
            self.canvas.image = photo  # Keep reference
            
            return True
        except Exception as e:
            logging.error(f"Error rendering image: {e}", exc_info=True)
            return False

if __name__ == "__main__":
    FractalGenerator()