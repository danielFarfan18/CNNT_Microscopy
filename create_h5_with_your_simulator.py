#!/usr/bin/env python3
"""
Script para crear archivos H5 optimizados usando tu DefocusSimulator existente
Este script te permite mantener tu flujo de trabajo actual pero optimizado para multi-sigma training

Modo 1: Solo imágenes limpias (recomendado para multi-sigma training)
Modo 2: Pares tradicionales (para compatibilidad con tu código actual)
Modo 3: Híbrido (ambas opciones)
"""

import h5py
import numpy as np
import cv2
import os
from pathlib import Path
import json
from tqdm import tqdm

# Importa tu DefocusSimulator existente
# from your_defocus_simulator import DefocusSimulator

class H5CreatorForCNNT:
    """
    Adaptador para crear archivos H5 compatibles con el nuevo sistema multi-sigma
    """
    
    def __init__(self, defocus_simulator=None):
        """
        @args:
            defocus_simulator: Tu DefocusSimulator existente (opcional)
        """
        self.defocus_simulator = defocus_simulator
        
    def create_h5_clean_only(self, image_dir, output_h5_path, key_prefix="sample"):
        """
        Modo 1: Crear H5 con solo imágenes limpias para multi-sigma training
        
        @args:
            image_dir: Directorio con imágenes originales/limpias
            output_h5_path: Ruta del archivo H5 de salida
            key_prefix: Prefijo para las claves en H5
        """
        image_files = self._get_image_files(image_dir)
        
        with h5py.File(output_h5_path, 'w') as h5f:
            for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
                try:
                    # Cargar imagen
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    
                    # Normalizar si es necesario
                    image = self._normalize_image(image)
                    
                    # Crear key única
                    key = f"{key_prefix}_{idx:06d}_{img_path.stem}"
                    
                    # Guardar solo imagen limpia
                    # El blur se generará dinámicamente durante el entrenamiento
                    h5f.create_dataset(f"{key}/clean_im", data=image, compression='gzip')
                    
                    # Opcional: crear una versión con blur fijo para validación/testing
                    if self.defocus_simulator:
                        # Usar tu simulador para crear UNA versión con blur moderado
                        blurred = self._apply_your_simulator(image, sigma=3.0)
                        h5f.create_dataset(f"{key}/noisy_im", data=blurred, compression='gzip')
                    else:
                        # Si no tienes simulador, usa la misma imagen limpia
                        h5f.create_dataset(f"{key}/noisy_im", data=image, compression='gzip')
                    
                    # Metadatos
                    h5f.attrs[f"{key}_source"] = str(img_path)
                    h5f.attrs[f"{key}_shape"] = image.shape
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        print(f"Created H5 file: {output_h5_path}")
        print(f"Mode: Clean images for multi-sigma training")
        print(f"Images processed: {len(image_files)}")
    
    def create_h5_traditional_pairs(self, image_dir, output_h5_path, 
                                  blur_configs, key_prefix="sample"):
        """
        Modo 2: Crear H5 con pares tradicionales usando tu DefocusSimulator
        Compatible con tu código actual
        
        @args:
            image_dir: Directorio con imágenes originales
            output_h5_path: Ruta del archivo H5 de salida
            blur_configs: Lista de configuraciones de blur
            key_prefix: Prefijo para las claves
        """
        if not self.defocus_simulator:
            raise ValueError("DefocusSimulator is required for traditional pairs mode")
        
        image_files = self._get_image_files(image_dir)
        
        with h5py.File(output_h5_path, 'w') as h5f:
            sample_idx = 0
            
            for img_path in tqdm(image_files, desc="Processing images"):
                try:
                    # Cargar imagen original
                    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if original is None:
                        continue
                    
                    original = self._normalize_image(original)
                    
                    # Crear múltiples versiones con diferentes blurs
                    for config_idx, blur_config in enumerate(blur_configs):
                        # Aplicar tu simulador con configuración específica
                        blurred_images, metadata = self._apply_your_simulator_multiple(
                            original, blur_config
                        )
                        
                        for blur_idx, blurred in enumerate(blurred_images):
                            key = f"{key_prefix}_{sample_idx:06d}"
                            
                            # Guardar par tradicional
                            h5f.create_dataset(f"{key}/noisy_im", data=blurred, compression='gzip')
                            h5f.create_dataset(f"{key}/clean_im", data=original, compression='gzip')
                            
                            # Metadatos
                            h5f.attrs[f"{key}_source"] = str(img_path)
                            h5f.attrs[f"{key}_config"] = str(blur_config)
                            h5f.attrs[f"{key}_blur_idx"] = blur_idx
                            
                            sample_idx += 1
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        print(f"Created H5 file: {output_h5_path}")
        print(f"Mode: Traditional pairs")
        print(f"Total samples: {sample_idx}")
    
    def create_h5_hybrid(self, image_dir, output_h5_path, key_prefix="sample"):
        """
        Modo 3: Híbrido - Imágenes limpias + algunos pares fijos para validación
        """
        image_files = self._get_image_files(image_dir)
        
        with h5py.File(output_h5_path, 'w') as h5f:
            for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
                try:
                    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if original is None:
                        continue
                    
                    original = self._normalize_image(original)
                    
                    # Imagen principal (limpia) para multi-sigma training
                    main_key = f"{key_prefix}_{idx:06d}_main"
                    h5f.create_dataset(f"{main_key}/clean_im", data=original, compression='gzip')
                    h5f.create_dataset(f"{main_key}/noisy_im", data=original, compression='gzip')  # Placeholder
                    
                    # Algunas versiones fijas para validación/testing (opcional)
                    if self.defocus_simulator and idx % 10 == 0:  # Solo cada 10 imágenes
                        for sigma in [1.0, 3.0, 5.0]:
                            val_key = f"{key_prefix}_{idx:06d}_val_sigma{sigma}"
                            blurred = self._apply_your_simulator(original, sigma=sigma)
                            h5f.create_dataset(f"{val_key}/clean_im", data=original, compression='gzip')
                            h5f.create_dataset(f"{val_key}/noisy_im", data=blurred, compression='gzip')
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
    
    def _get_image_files(self, image_dir):
        """Obtener archivos de imagen del directorio"""
        supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        image_files = []
        
        for fmt in supported_formats:
            image_files.extend(Path(image_dir).glob(f"*{fmt}"))
            image_files.extend(Path(image_dir).glob(f"*{fmt.upper()}"))
        
        return sorted(image_files)
    
    def _normalize_image(self, image):
        """Normalizar imagen (adapta según tus necesidades)"""
        # Opción 1: Mantener valores originales
        return image.astype(np.float32)
        
        # Opción 2: Normalizar a [0, 1]
        # return (image.astype(np.float32) / 255.0)
        
        # Opción 3: Normalizar por percentiles
        # p1, p99 = np.percentile(image, [1, 99])
        # return np.clip((image - p1) / (p99 - p1), 0, 1).astype(np.float32)
    
    def _apply_your_simulator(self, image, sigma=3.0):
        """
        Aplicar tu DefocusSimulator para un sigma específico
        Adapta esta función a tu implementación
        """
        if not self.defocus_simulator:
            # Fallback simple con OpenCV
            kernel_size = max(3, int(2 * np.ceil(2 * sigma) + 1))
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        # Usar tu simulador
        # Ejemplo de adaptación (ajusta según tu API):
        try:
            # Si tu simulador puede tomar sigma directamente
            return self.defocus_simulator.apply_blur_with_sigma(image, sigma)
        except:
            # Fallback
            kernel_size = max(3, int(2 * np.ceil(2 * sigma) + 1))
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def _apply_your_simulator_multiple(self, image, blur_config):
        """
        Aplicar múltiples configuraciones de blur usando tu simulador
        """
        if not self.defocus_simulator:
            # Fallback simple
            results = []
            for sigma in [1.0, 3.0, 5.0]:
                blurred = self._apply_your_simulator(image, sigma)
                results.append(blurred)
            return results, {}
        
        # Usar tu simulador completo
        # Adapta según tu API:
        try:
            return self.defocus_simulator.process_single_image(image, blur_config)
        except:
            # Fallback
            return [self._apply_your_simulator(image, 3.0)], {}


def main():
    """Ejemplo de uso"""
    
    # Configuración
    image_dir = "path/to/your/clean/images"
    output_dir = "path/to/h5/output"
    
    # Inicializar tu simulador (opcional)
    # defocus_sim = DefocusSimulator()  # Tu clase existente
    defocus_sim = None  # Por ahora sin simulador
    
    # Crear el adaptador
    h5_creator = H5CreatorForCNNT(defocus_sim)
    
    print("=== Creando archivos H5 para CNNT Multi-Sigma ===\n")
    
    # MODO 1: Solo imágenes limpias (RECOMENDADO para multi-sigma)
    print("1. Creando H5 con imágenes limpias (para multi-sigma training)...")
    h5_creator.create_h5_clean_only(
        image_dir=image_dir,
        output_h5_path=f"{output_dir}/clean_images_for_multisigma.h5"
    )
    
    # MODO 2: Pares tradicionales (para compatibilidad)
    if defocus_sim:
        print("\n2. Creando H5 con pares tradicionales...")
        blur_configs = [
            {"max_sigma": 2, "num_samples": 2},
            {"max_sigma": 5, "num_samples": 3},
            {"max_sigma": 8, "num_samples": 2},
        ]
        h5_creator.create_h5_traditional_pairs(
            image_dir=image_dir,
            output_h5_path=f"{output_dir}/traditional_pairs.h5",
            blur_configs=blur_configs
        )
    
    # MODO 3: Híbrido
    print("\n3. Creando H5 híbrido...")
    h5_creator.create_h5_hybrid(
        image_dir=image_dir,
        output_h5_path=f"{output_dir}/hybrid_dataset.h5"
    )
    
    print("\n=== Cómo usar estos archivos ===")
    print("Para multi-sigma training:")
    print(f"python main.py --h5files {output_dir}/clean_images_for_multisigma.h5 --enable_blur_simulation")
    print("\nPara entrenamiento tradicional:")
    print(f"python main.py --h5files {output_dir}/traditional_pairs.h5")


if __name__ == "__main__":
    main()