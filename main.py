import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import random
from datetime import datetime
import time  # Añadido import de time
import logging
import os
import json  # Import json for configuration saving
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage  # Moved import here for accessibility

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
        self.window.geometry("1300x900")
        
        # Agregar dimensiones base
        self.width = 1280
        self.height = 720
        
        self.current_image = None
        self.current_fractal_info = {}
        
        # Create button frames for better organization
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=10)
        
        # Create label for fractal selection
        ttk.Label(button_frame, text="Fractal Type:").pack(side=tk.LEFT, padx=5)
        
        # Agregar valor inicial para C
        self.C = complex(-0.4, 0.6)  # Valor inicial para conjunto Julia
        
        # Create buttons for different fractals
        self.julia_btn = ttk.Button(button_frame, text="Julia Set", 
                               command=lambda: self.generate_fractal('julia'))
        self.julia_btn.pack(side=tk.LEFT, padx=5)
        
        self.mandelbrot_btn = ttk.Button(button_frame, text="Mandelbrot Set", 
                                    command=lambda: self.generate_fractal('mandelbrot'))
        self.mandelbrot_btn.pack(side=tk.LEFT, padx=5)
        
        self.burning_ship_btn = ttk.Button(button_frame, text="Burning Ship", 
                                      command=lambda: self.generate_fractal('burning_ship'))
        self.burning_ship_btn.pack(side=tk.LEFT, padx=5)
        
        # Create save button
        self.save_btn = ttk.Button(button_frame, text="Save HD Image", command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=20)  # Added more padding to separate from fractal buttons

        # Create progress bar
        self.progress = ttk.Progressbar(self.window, length=300, mode='determinate')
        self.progress.pack(pady=5)

        # Create canvas for displaying the fractal
        self.canvas = tk.Canvas(self.window, width=1280, height=720)
        self.canvas.pack(pady=10)

        # Create text area for fractal information
        self.info_text = tk.Text(self.window, height=15, width=80)  # Increased height from 5 to 15
        self.info_text.pack(pady=10)

        self.fractal_cache = {}  # Add cache dictionary
        self.max_cache_size = 5  # Keep last 5 fractals in memory

        self.generate_fractal()
        self.create_menu()
        self.window.mainloop()

    def create_menu(self):
        """Create menu bar with additional options"""
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Current", command=self.save_image)
        file_menu.add_command(label="Save Preferences", command=self.save_preferences)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.window.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Info Panel", command=self.toggle_info)

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
Symmetry: {abs(c_value.imag) < 0.0001 and "Real Axis" or "Complex Plane"}"""

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

            else:  # Burning Ship
                math_formula = """Mathematical Formula:
zₙ₊₁ = (|Re(zₙ)| + |Im(zₙ)|i)² + c
where z₀ = 0
Escape Criterion: |z| > 2
Domain: Complex plane c"""

                classification = """Burning Ship Classification:
Variation of the Mandelbrot set using absolute values
Creates flame-like structures and a distinctive 'ship' shape"""

            # Color algorithm description
            color_info = f"""Color Algorithm:
RGB = (sin(s·r_m/30 + φr)·127 + 128, sin(s·g_m/30 + φg)·127 + 128, sin(s·b_m/30 + φb)·127 + 128)
where s = iteration + 1 - log₂(log₂(|z|))
{color_info}
{phase_info}"""

            # Technical parameters
            tech_info = f"""Technical Parameters:
Resolution: {self.canvas.winfo_width()}x{self.canvas.winfo_height()}
Maximum Iterations: {info.get('iterations', 'N/A')}
Generation Time: {info.get('time', 'N/A')} seconds
{parameters}"""

            # Combine all information
            full_info = f"""=== {fractal_type} Technical Information ===

{math_formula}

{classification}

{color_info}

{tech_info}"""
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, full_info)
            
        except Exception as e:
            error_info = f"Error displaying information: {str(e)}"
            print(f"Debug - Error details: {e}")  # For debugging
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, error_info)

    def generate_fractal_chunk(self, params):
        """Generate a portion of the fractal"""
        start_row, end_row, width, height, C, color_mults, color_phases = params
        chunk = np.zeros((end_row - start_row, width, 3), dtype=np.uint8)
        
        # Calculate boundaries
        x = np.linspace(-1.5, 1.5, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y[start_row:end_row])
        Z = X + Y*1j
        
        # Usar los parámetros de color recibidos
        r_mult, g_mult, b_mult = color_mults
        phase_r, phase_g, phase_b = color_phases
        
        iteration_count = np.zeros_like(Z, dtype=int)
        max_iter = 1000
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C
            iteration_count[mask] = i
    
        log_zn = np.log2(np.abs(Z))
        smooth_iter = iteration_count + 1 - np.log2(log_zn)
    
        # Apply coloring
        chunk[:,:,0] = np.sin(smooth_iter * r_mult/30 + phase_r) * 127 + 128
        chunk[:,:,1] = np.sin(smooth_iter * g_mult/30 + phase_g) * 127 + 128
        chunk[:,:,2] = np.sin(smooth_iter * b_mult/30 + phase_b) * 127 + 128
    
        # Ensure no pure black areas
        dark_mask = np.all(chunk < 30, axis=2)
        chunk[dark_mask] = [30, 30, 30]
    
        return chunk

    def generate_julia(self):
        start_time = time.time()
    
        # Interesting Julia set parameters and regions
        julia_regions = [
            {'C': complex(-0.4, 0.6), 'center': (0, 0), 'zoom': 1},  # Dragon-like
            {'C': complex(0.285, 0), 'center': (0.2, 0.2), 'zoom': 2},  # Dendrite
            {'C': complex(0.45, 0.1428), 'center': (-0.3, 0), 'zoom': 1.5},  # Exotic Spiral
            {'C': complex(-0.70176, -0.3842), 'center': (0.1, -0.2), 'zoom': 2.5},  # Spiraling Tendrils
            {'C': complex(-0.835, -0.2321), 'center': (-0.1, 0.3), 'zoom': 1.8},  # Delicate Branches
            {'C': complex(0.35, 0.35), 'center': (0, -0.1), 'zoom': 2.2},  # Symmetric Pattern
            {'C': complex(0, 0.8), 'center': (0.15, 0.15), 'zoom': 1.7},  # Rabbit-like
            {'C': complex(-0.123, 0.745), 'center': (-0.2, 0.1), 'zoom': 2.3},  # Douady's Rabbit
            {'C': complex(-0.391, -0.587), 'center': (0.1, 0.1), 'zoom': 1.9},  # Spiral Galaxy
            {'C': complex(-0.54, 0.54), 'center': (-0.1, -0.1), 'zoom': 2.1}  # Exotic Flower
        ]
        
        # Random selection of region and parameters
        region = random.choice(julia_regions)
        self.C = region['C']
        center_x, center_y = region['center']
        base_zoom = region['zoom']
        zoom = base_zoom * random.uniform(0.5, 2.5)
        
        # Calculate boundaries based on zoom and center
        x_range = 3.0 / zoom
        y_range = x_range * self.height / self.width
        x = np.linspace(center_x - x_range/2, center_x + x_range/2, self.width)
        y = np.linspace(center_y - y_range/2, center_y + y_range/2, self.height)
        
        # Split into chunks for parallel processing
        chunks = 4
        chunk_size = self.height // chunks
        self.update_progress(0)
        
        # Generate color parameters once for all chunks
        r_mult = random.randint(3, 12)
        g_mult = random.randint(3, 12)
        b_mult = random.randint(3, 12)
        phase_r = random.random() * 2 * np.pi
        phase_g = random.random() * 2 * np.pi
        phase_b = random.random() * 2 * np.pi
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(chunks):
                start_row = i * chunk_size
                end_row = start_row + chunk_size if i < chunks - 1 else self.height
                # Pasar los parámetros de color a cada chunk
                params = (start_row, end_row, self.width, self.height, self.C, 
                     (r_mult, g_mult, b_mult), (phase_r, phase_g, phase_b))
                futures.append(executor.submit(self.generate_fractal_chunk, params))
                self.update_progress((i + 1) * 25)
            
            image = np.vstack([f.result() for f in futures])
    
        self.update_progress(100)
        
        # Add fractal information with color parameters
        self.current_fractal_info = {
            'type': 'Julia Set',
            'parameters': f'C: {self.C}, Center: ({center_x:.4f}, {center_y:.4f}), Zoom: {zoom:.1f}x',
            'iterations': 1000,
            'color_info': f'RGB multipliers: ({r_mult}, {g_mult}, {b_mult})',
            'phase_info': f'Phase shifts: ({phase_r:.4f}, {phase_g:.4f}, {phase_b:.4f})',
            'time': f"{time.time() - start_time:.2f}"
        }
    
        return image

    def generate_mandelbrot(self):
        start_time = time.time()
        
        width, height = 1280, 720
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
        
        # Enhanced color parameters
        r_mult = random.randint(3, 12)
        g_mult = random.randint(3, 12)
        b_mult = random.randint(3, 12)
        
        # Phase shifts for smooth color transitions
        phase_r = random.random() * 2 * np.pi
        phase_g = random.random() * 2 * np.pi
        phase_b = random.random() * 2 * np.pi
        
        iteration_count = np.zeros_like(Z, dtype=int)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            iteration_count[mask] = i
            
            if i % 50 == 0:
                self.update_progress((i * 100) // max_iter)
        
        log_zn = np.log2(np.abs(Z))
        smooth_iter = iteration_count + 1 - np.log2(log_zn)
        
        image[:,:,0] = np.sin(smooth_iter * r_mult/30 + phase_r) * 127 + 128
        image[:,:,1] = np.sin(smooth_iter * g_mult/30 + phase_g) * 127 + 128
        image[:,:,2] = np.sin(smooth_iter * b_mult/30 + phase_b) * 127 + 128
        
        dark_mask = np.all(image < 30, axis=2)
        image[dark_mask] = [30, 30, 30]
        
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
        
        width, height = 1280, 720
        max_iter = 1000
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Interesting regions in the Burning Ship fractal
        regions = [
            {'center': (-1.4, -0.1), 'zoom': 50},  # Main ship
            {'center': (-1.7, -0.028), 'zoom': 100},  # Detailed hull
            {'center': (-1.75, -0.04), 'zoom': 150},  # Side structures
            {'center': (-1.65, -0.02), 'zoom': 200},  # Small ships
            {'center': (-1.8, -0.08), 'zoom': 120},  # Bridge section
            {'center': (-1.73, -0.01), 'zoom': 180},  # Antenna detail
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
        
        r_mult = random.randint(3, 12)
        g_mult = random.randint(3, 12)
        b_mult = random.randint(3, 12)
        
        phase_r = random.random() * 2 * np.pi
        phase_g = random.random() * 2 * np.pi
        phase_b = random.random() * 2 * np.pi
        
        iteration_count = np.zeros_like(Z, dtype=int)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = (np.abs(Z[mask].real) + np.abs(Z[mask].imag)*1j)**2 + C[mask]
            iteration_count[mask] = i
            
            if i % 50 == 0:
                self.update_progress((i * 100) // max_iter)
        
        log_zn = np.log2(np.abs(Z))
        smooth_iter = iteration_count + 1 - np.log2(log_zn)
        
        image[:,:,0] = np.sin(smooth_iter * r_mult/30 + phase_r) * 127 + 128
        image[:,:,1] = np.sin(smooth_iter * g_mult/30 + phase_g) * 127 + 128
        image[:,:,2] = np.sin(smooth_iter * b_mult/30 + phase_b) * 127 + 128
        
        dark_mask = np.all(image < 30, axis=2)
        image[dark_mask] = [30, 30, 30]
        
        self.current_fractal_info = {
            'type': 'Burning Ship Fractal',
            'parameters': f'Center: ({center_x:.4f}, {center_y:.4f}), Zoom: {zoom:.1f}x',
            'iterations': max_iter,
            'color_info': f'RGB multipliers: ({r_mult}, {g_mult}, {b_mult})',
            'phase_info': f'Phase shifts: ({phase_r:.4f}, {phase_g:.4f}, {phase_b:.4f})',
            'time': f"{time.time() - start_time:.2f}"
        }
        
        return image

    def generate_fractal(self, fractal_type='julia'):
        # Disable all buttons during generation
        self.disable_buttons()
        self.progress['value'] = 0
        self.window.update_idletasks()
        
        try:
            self.update_progress(0)
            
            # Generar el fractal
            if fractal_type == 'julia':
                image_array = self.generate_julia()
            elif fractal_type == 'mandelbrot':
                image_array = self.generate_mandelbrot()
            else:
                image_array = self.generate_burning_ship()
                
            self.update_progress(75)
            
            # Aplicar mejoras de calidad
            image_array = self.adjust_image_quality(image_array)
            
            # Renderizar la imagen
            if self.render_image(image_array):
                self.update_info()
                self.update_progress(100)
            else:
                raise Exception("Failed to render image")
                
        except Exception as e:
            print(f"Error generating fractal: {e}")
            logging.error(f"Error generating fractal: {e}", exc_info=True)
        finally:
            self.enable_buttons()

    def disable_buttons(self):
        """Deshabilitar todos los botones"""
        for button in [self.julia_btn, self.mandelbrot_btn, 
                      self.burning_ship_btn, self.save_btn]:
            button['state'] = 'disabled'

    def enable_buttons(self):
        """Habilitar todos los botones"""
        for button in [self.julia_btn, self.mandelbrot_btn, 
                      self.burning_ship_btn, self.save_btn]:
            button['state'] = 'normal'

    def save_image(self):
        """Guarda el último fractal generado en alta resolución"""
        if not self.current_fractal_info or not self.current_image:
            print("No fractal generated to save")
            return
            
        self.save_btn['state'] = 'disabled'
        try:
            # Crear directorio si no existe
            if not os.path.exists('fractals'):
                os.makedirs('fractals')
            
            # Obtener información del fractal actual
            fractal_type = self.current_fractal_info['type']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generar nombre de archivo
            filename = f"fractals/{fractal_type.replace(' ', '_')}_{timestamp}_4K.png"
            
            # Redimensionar a 4K
            image_4k = self.current_image.resize((3840, 2160), Image.Resampling.LANCZOS)
            
            # Aplicar mejoras de calidad
            image_array = np.array(image_4k)
            enhanced_image = self.adjust_image_quality(image_array)
            
            # Guardar imagen
            Image.fromarray(enhanced_image).save(filename, "PNG", optimize=True)
            
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
        
        # Apply subtle gaussian blur for smoother gradients
        image = ndimage.gaussian_filter(image, sigma=[0.5, 0.5, 0])
        
        # Enhance contrast
        p2, p98 = np.percentile(image, (2, 98))
        image = np.clip(image, p2, p98)
        image = ((image - p2) / (p98 - p2) * 255).astype(np.uint8)
        
        return image

    def toggle_info(self):
        """Toggle visibility of info panel"""
        if self.info_text.winfo_viewable():
            self.info_text.pack_forget()
        else:
            self.info_text.pack(pady=10)

    def render_image(self, image_array):
        """Método centralizado para renderizar imágenes"""
        try:
            # Asegurarse de que la imagen esté en el formato correcto
            image_array = image_array.astype(np.uint8)
            
            # Convertir numpy array a PIL Image
            image = Image.fromarray(image_array)
            
            # Redimensionar si es necesario para ajustarse al canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if image.size != (canvas_width, canvas_height):
                image = image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # Guardar referencia a la imagen actual
            self.current_image = image
            
            # Convertir a PhotoImage y mostrar
            photo = ImageTk.PhotoImage(image)
            self.canvas.delete("all")  # Limpiar canvas antes de mostrar nueva imagen
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
            self.canvas.image = photo  # Mantener referencia
            
            return True
        except Exception as e:
            logging.error(f"Error rendering image: {e}", exc_info=True)
            return False

if __name__ == "__main__":
    FractalGenerator()