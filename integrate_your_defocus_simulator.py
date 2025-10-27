#!/usr/bin/env python3
"""
Integración específica de tu DefocusSimulator con el nuevo sistema multi-sigma

Este script muestra cómo adaptar tu código existente para trabajar con la nueva implementación
manteniendo tu flujo de trabajo pero mejorando los resultados de entrenamiento.
"""

import numpy as np
import cv2
import h5py
from pathlib import Path
import json
from tqdm import tqdm

# PASO 1: Adaptar tu DefocusSimulator existente
class YourDefocusSimulatorAdapter:
    """
    Adaptador que integra tu DefocusSimulator existente con el nuevo sistema
    """
    
    def __init__(self):
        """Inicializa con tu simulador existente"""
        # Importa y usa tu DefocusSimulator existente
        # from your_file import DefocusSimulator
        # self.original_simulator = DefocusSimulator()
        self.original_simulator = None  # Por ahora placeholder
        
    def apply_single_sigma_blur(self, image, sigma):
        """
        Aplica blur con un solo valor de sigma
        Adapta esta función a tu DefocusSimulator
        """
        if sigma <= 0:
            return image.copy()
            
        # Método simple usando OpenCV (reemplaza con tu implementación)
        kernel_size = max(3, int(2 * np.ceil(2 * sigma) + 1))
        kernel_size = min(kernel_size, 15)  # Límite razonable
        
        if kernel_size % 2 == 0:  # Asegurar que sea impar
            kernel_size += 1
            
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def apply_multi_sigma_regions(self, image, sigma_regions):
        """
        Aplica blur con diferentes sigmas en diferentes regiones
        Esto adapta tu lógica de regiones múltiples
        
        @args:
            image: Imagen de entrada
            sigma_regions: Lista de dict con {'center', 'size', 'sigma', 'intensity'}
        """
        result = image.copy().astype(np.float32)
        height, width = image.shape
        
        # Crear máscara acumulativa
        total_mask = np.zeros((height, width), dtype=np.float32)
        blurred_accumulator = np.zeros((height, width), dtype=np.float32)
        
        for region in sigma_regions:
            center_x, center_y = region['center']
            region_size = region['size']
            sigma = region['sigma']
            intensity = region.get('intensity', 1.0)
            
            # Crear máscara para esta región (tu lógica existente)
            transition_width = region_size * 0.2
            y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
            dist = np.sqrt(x*x + y*y)
            region_mask = 1 / (1 + np.exp((dist - region_size/2) / transition_width))
            region_mask *= intensity
            
            # Aplicar blur específico para esta región
            blurred_region = self.apply_single_sigma_blur(image, sigma)
            
            # Acumular en el resultado
            blurred_accumulator += blurred_region * region_mask
            total_mask += region_mask
        
        # Combinar con imagen original
        final_mask = np.clip(total_mask, 0, 1)
        result = image * (1 - final_mask) + blurred_accumulator * (final_mask > 0) / np.maximum(total_mask, 1e-6)
        
        return np.clip(result, 0, 255).astype(image.dtype)
    
    def create_progressive_blur_sample(self, image, difficulty_level=1, stage_progress=0.0):
        """
        Crea una muestra con blur progresivo basado en tu lógica
        
        @args:
            image: Imagen limpia de entrada
            difficulty_level: 0=fácil, 1=medio, 2=difícil, 3=extremo
            stage_progress: 0.0-1.0, progreso dentro del stage
        """
        height, width = image.shape
        
        # Configuraciones progresivas (adapta según tu experiencia)
        configs = [
            {  # Fácil
                'sigma_range': (0.5, 2.0),
                'num_regions': (1, 2),
                'region_size_range': (80, 150),
                'overlap_prob': 0.0
            },
            {  # Medio
                'sigma_range': (1.0, 3.5),
                'num_regions': (2, 3),
                'region_size_range': (60, 180),
                'overlap_prob': 0.3
            },
            {  # Difícil
                'sigma_range': (2.0, 5.0),
                'num_regions': (2, 4),
                'region_size_range': (50, 200),
                'overlap_prob': 0.5
            },
            {  # Extremo
                'sigma_range': (3.0, 8.0),
                'num_regions': (3, 5),
                'region_size_range': (40, 220),
                'overlap_prob': 0.7
            }
        ]
        
        config = configs[min(difficulty_level, 3)]
        
        # Generar regiones
        num_regions = np.random.randint(config['num_regions'][0], config['num_regions'][1] + 1)
        sigma_regions = []
        
        for i in range(num_regions):
            # Sigma progresivo dentro del rango
            sigma_min, sigma_max = config['sigma_range']
            sigma = np.random.uniform(sigma_min, sigma_max)
            
            # Tamaño de región
            size_min, size_max = config['region_size_range']
            region_size = np.random.randint(size_min, size_max)
            
            # Posición (con validación)
            max_attempts = 20
            for attempt in range(max_attempts):
                center_x = np.random.randint(region_size//2, width - region_size//2)
                center_y = np.random.randint(region_size//2, height - region_size//2)
                
                # Verificar overlap si es necesario
                if np.random.random() > config['overlap_prob']:
                    # Verificar que no se superponga demasiado
                    valid = True
                    for existing in sigma_regions:
                        dist = np.sqrt((center_x - existing['center'][0])**2 + 
                                     (center_y - existing['center'][1])**2)
                        if dist < (region_size + existing['size']) * 0.4:
                            valid = False
                            break
                    if not valid:
                        continue
                
                # Región válida
                sigma_regions.append({
                    'center': (center_x, center_y),
                    'size': region_size,
                    'sigma': sigma,
                    'intensity': 1.0
                })
                break
        
        # Aplicar blur con regiones múltiples
        if len(sigma_regions) > 0:
            blurred = self.apply_multi_sigma_regions(image, sigma_regions)
        else:
            blurred = image.copy()
        
        # Información de la muestra
        blur_info = {
            'difficulty_level': difficulty_level,
            'stage_progress': stage_progress,
            'num_regions': len(sigma_regions),
            'sigma_ranges_used': [r['sigma'] for r in sigma_regions],
            'avg_sigma': np.mean([r['sigma'] for r in sigma_regions]) if sigma_regions else 0,
            'max_sigma': np.max([r['sigma'] for r in sigma_regions]) if sigma_regions else 0,
            'regions': sigma_regions
        }
        
        return blurred, blur_info


# PASO 2: Crear archivos H5 usando tu simulador adaptado
def create_h5_with_your_simulator(input_dir, output_h5_path, mode='progressive'):
    """
    Crea archivos H5 usando tu simulador adaptado
    
    @args:
        input_dir: Directorio con imágenes originales
        output_h5_path: Ruta de salida del H5
        mode: 'progressive' para multi-sigma, 'traditional' para pares fijos
    """
    simulator = YourDefocusSimulatorAdapter()
    image_files = list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.tif"))
    
    with h5py.File(output_h5_path, 'w') as h5f:
        if mode == 'progressive':
            # MODO PROGRESIVO: Solo imágenes limpias + metadatos
            print("Creando dataset para multi-sigma training...")
            
            for idx, img_path in enumerate(tqdm(image_files)):
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                # Normalizar
                image = image.astype(np.float32)
                
                # Clave única
                key = f"sample_{idx:06d}_{img_path.stem}"
                
                # Guardar imagen limpia (principal)
                h5f.create_dataset(f"{key}/clean_im", data=image, compression='gzip')
                
                # Crear una versión de referencia con blur moderado para validación
                ref_blurred, ref_info = simulator.create_progressive_blur_sample(
                    image, difficulty_level=1, stage_progress=0.5
                )
                h5f.create_dataset(f"{key}/noisy_im", data=ref_blurred, compression='gzip')
                
                # Metadatos
                h5f.attrs[f"{key}_source"] = str(img_path)
                h5f.attrs[f"{key}_ref_blur_info"] = json.dumps(ref_info)
        
        elif mode == 'traditional':
            # MODO TRADICIONAL: Múltiples pares fijos
            print("Creando dataset tradicional con múltiples dificultades...")
            
            sample_idx = 0
            for img_path in tqdm(image_files):
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                image = image.astype(np.float32)
                
                # Crear múltiples versiones con diferentes dificultades
                for difficulty in range(4):  # 0=fácil, 3=extremo
                    for sample in range(2):  # 2 muestras por dificultad
                        blurred, blur_info = simulator.create_progressive_blur_sample(
                            image, difficulty_level=difficulty, stage_progress=0.5
                        )
                        
                        key = f"sample_{sample_idx:06d}"
                        h5f.create_dataset(f"{key}/clean_im", data=image, compression='gzip')
                        h5f.create_dataset(f"{key}/noisy_im", data=blurred, compression='gzip')
                        
                        # Metadatos
                        h5f.attrs[f"{key}_source"] = str(img_path)
                        h5f.attrs[f"{key}_difficulty"] = difficulty
                        h5f.attrs[f"{key}_blur_info"] = json.dumps(blur_info)
                        
                        sample_idx += 1
    
    print(f"Dataset creado: {output_h5_path}")
    print(f"Modo: {mode}")


# PASO 3: Ejemplo de uso completo
def main():
    """Ejemplo completo de integración"""
    
    print("=== Integración de tu DefocusSimulator con Multi-Sigma Training ===\n")
    
    # Rutas (ajusta según tu estructura)
    input_images_dir = "dataset_sea_urchin/train"  # Tu directorio de imágenes originales
    output_dir = "h5_datasets"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Opción 1: Dataset progresivo (RECOMENDADO)
    print("1. Creando dataset progresivo para multi-sigma training...")
    progressive_h5 = f"{output_dir}/progressive_multisigma.h5"
    create_h5_with_your_simulator(
        input_dir=input_images_dir,
        output_h5_path=progressive_h5,
        mode='progressive'
    )
    
    # Opción 2: Dataset tradicional (para comparación)
    print("\n2. Creando dataset tradicional...")
    traditional_h5 = f"{output_dir}/traditional_pairs.h5"
    create_h5_with_your_simulator(
        input_dir=input_images_dir,
        output_h5_path=traditional_h5,
        mode='traditional'
    )
    
    print("\n=== Cómo entrenar ===")
    print("Multi-sigma training (recomendado):")
    print(f"python main.py --h5files {progressive_h5} --enable_blur_simulation --num_epochs 50")
    
    print("\nTraining tradicional:")
    print(f"python main.py --h5files {traditional_h5} --num_epochs 50")
    
    print("\nMulti-sigma con configuración personalizada:")
    print(f"python main.py --h5files {progressive_h5} --enable_blur_simulation \\")
    print("    --blur_sigma_ranges '0.5,2.0' '1.0,3.5' '2.0,5.0' '3.0,8.0' \\")
    print("    --blur_probabilities 0.4 0.3 0.2 0.1 \\")
    print("    --variable_blur_prob 0.4 \\")  # Mayor porque ya tienes experiencia con regiones
    print("    --num_epochs 50")
    
    print("\n=== Ventajas de esta integración ===")
    print("✅ Mantiene tu lógica de simulación de blur")
    print("✅ Añade curriculum learning automático")
    print("✅ Mejora la generalización a diferentes sigmas")
    print("✅ Compatible con tu flujo de trabajo existente")
    print("✅ Evita el problema de emborrona las imágenes")


# PASO 4: Función para migrar tus datasets existentes
def migrate_existing_h5_to_multisigma(existing_h5_path, new_h5_path):
    """
    Migra un archivo H5 existente al formato multi-sigma
    
    @args:
        existing_h5_path: Tu archivo H5 actual con pares (blur, clean)
        new_h5_path: Nuevo archivo H5 optimizado para multi-sigma
    """
    print(f"Migrando {existing_h5_path} a formato multi-sigma...")
    
    with h5py.File(existing_h5_path, 'r') as old_h5:
        with h5py.File(new_h5_path, 'w') as new_h5:
            
            processed_keys = set()
            sample_idx = 0
            
            for key in old_h5.keys():
                # Extraer el nombre base (sin sufijos como _defocus_1)
                base_name = key.split('_defocus')[0] if '_defocus' in key else key
                
                if base_name in processed_keys:
                    continue
                processed_keys.add(base_name)
                
                try:
                    # Buscar imagen limpia
                    clean_data = None
                    if f"{key}/clean_im" in old_h5:
                        clean_data = old_h5[f"{key}/clean_im"][...]
                    elif f"{base_name}/clean_im" in old_h5:
                        clean_data = old_h5[f"{base_name}/clean_im"][...]
                    
                    if clean_data is not None:
                        new_key = f"sample_{sample_idx:06d}_{base_name}"
                        
                        # Guardar solo imagen limpia
                        new_h5.create_dataset(f"{new_key}/clean_im", data=clean_data, compression='gzip')
                        
                        # Crear placeholder para noisy (se generará dinámicamente)
                        new_h5.create_dataset(f"{new_key}/noisy_im", data=clean_data, compression='gzip')
                        
                        # Metadatos
                        new_h5.attrs[f"{new_key}_migrated_from"] = key
                        new_h5.attrs[f"{new_key}_original_file"] = existing_h5_path
                        
                        sample_idx += 1
                
                except Exception as e:
                    print(f"Error procesando {key}: {e}")
                    continue
    
    print(f"Migración completada: {sample_idx} muestras")
    print(f"Nuevo archivo: {new_h5_path}")
    print("Ahora puedes entrenar con: python main.py --h5files {} --enable_blur_simulation".format(new_h5_path))


if __name__ == "__main__":
    # Ejecutar ejemplo principal
    main()
    
    # Ejemplo de migración (descomenta si tienes archivos H5 existentes)
    # migrate_existing_h5_to_multisigma(
    #     existing_h5_path="partial_defocus_sigma_max_5/train/existing_dataset.h5",
    #     new_h5_path="h5_datasets/migrated_multisigma.h5"
    # )