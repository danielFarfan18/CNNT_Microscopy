# Multi-Sigma Blur Simulation for CNNT

## Problema Identificado

Tu implementación actual funciona bien para desenfoques uniformes con sigma=5, pero tiene dificultades con desenfoques de diferentes grados porque:

1. **Falta de variabilidad en el entrenamiento**: El dataset no incluía simulación de blur con diferentes sigmas
2. **Arquitectura no adaptativa**: La CNN no estaba diseñada para manejar diferentes grados de desenfoque
3. **Pérdida de información**: Para desenfoques severos, la información se pierde y el modelo tiende a generar artifacts

## Solución Implementada

### 1. **Simulador Avanzado de Desenfoque** (`AdvancedDefocusSimulator`)

- **Multi-sigma regions**: Cada región puede tener un sigma diferente
- **Curriculum learning**: Progresión automática de dificultad durante el entrenamiento
- **Blur espacialmente variable**: Simula condiciones reales de microscopía
- **Configuración flexible**: Fácil personalización para diferentes tipos de microscopía

### 2. **Curriculum Learning**

El modelo aprende progresivamente:
- **Época 0-25%**: Principalmente blur fácil (sigma 0.5-2.0)
- **Época 25-50%**: Combinación fácil-medio (sigma 0.5-4.0)
- **Época 50-75%**: Combinación medio-difícil (sigma 1.5-6.0)
- **Época 75-100%**: Todos los niveles incluyendo extremo (sigma 2.0-8.0)

### 3. **Integración con tu Dataset**

- Se integra seamlessly con tu `MicroscopyDataset`
- Usa tus imágenes limpias y genera blur dinámicamente
- Mantiene compatibilidad con tu pipeline existente

## Uso

### Entrenamiento Básico con Curriculum Learning (Recomendado)

```bash
python main.py \
    --h5files your_data.h5 \
    --enable_blur_simulation \
    --project CNNT_MultiSigma \
    --num_epochs 50 \
    --batch_size 4
```

### Entrenamiento con Dificultad Fija

```bash
# Fácil (para pruebas iniciales)
python main.py --enable_blur_simulation --blur_difficulty easy --h5files your_data.h5

# Medio
python main.py --enable_blur_simulation --blur_difficulty medium --h5files your_data.h5

# Difícil
python main.py --enable_blur_simulation --blur_difficulty hard --h5files your_data.h5

# Extremo
python main.py --enable_blur_simulation --blur_difficulty extreme --h5files your_data.h5
```

### Configuración Personalizada

```bash
python main.py \
    --enable_blur_simulation \
    --blur_sigma_ranges "0.5,1.5" "1.5,3.0" "3.0,5.0" "5.0,8.0" \
    --blur_probabilities 0.4 0.3 0.2 0.1 \
    --variable_blur_prob 0.3 \
    --no_blur_prob 0.1 \
    --h5files your_data.h5
```

## Parámetros de Configuración

### Argumentos Principales

- `--enable_blur_simulation`: Habilita la simulación de blur
- `--blur_difficulty`: Nivel fijo de dificultad (`easy`, `medium`, `hard`, `extreme`, o `None` para curriculum)
- `--blur_sigma_ranges`: Rangos de sigma para diferentes niveles (formato: "min,max")
- `--blur_probabilities`: Probabilidades de cada nivel de dificultad
- `--variable_blur_prob`: Probabilidad de usar blur espacialmente variable (0.0-1.0)
- `--no_blur_prob`: Probabilidad de usar imagen limpia como entrada (0.0-1.0)

### Configuraciones Recomendadas por Tipo de Microscopía

#### Microscopía de Luz
```bash
--blur_sigma_ranges "0.2,0.8" "0.8,1.5" "1.5,2.5" "2.5,4.0" \
--blur_probabilities 0.5 0.3 0.15 0.05
```

#### Microscopía Electrónica
```bash
--blur_sigma_ranges "1.0,2.0" "2.0,4.0" "4.0,6.0" "6.0,10.0" \
--blur_probabilities 0.3 0.4 0.2 0.1
```

## Mejoras en la Arquitectura

### Función de Pérdida Optimizada

```bash
--loss mse ssim sobel \
--loss_weights 0.3 1.0 0.2
```

- **MSE (0.3)**: Preservación de información básica
- **SSIM (1.0)**: Calidad perceptual principal
- **Sobel (0.2)**: Preservación de bordes

### Configuración de Optimización

```bash
--optim adamw \
--global_lr 3e-4 \
--scheduler OneCycleLR \
--weight_decay 0.01 \
--clip_grad_norm 1.0
```

## Script de Ejemplo

Ejecuta el script de ejemplo para ver todas las opciones:

```bash
python example_multi_sigma_training.py
```

## Ventajas de esta Implementación

1. **Mejor Generalización**: El modelo aprende a manejar múltiples niveles de desenfoque
2. **Curriculum Learning**: Previene overfitting y mejora la convergencia
3. **Blur Real**: Simula condiciones espacialmente variables como en microscopía real
4. **Flexibilidad**: Fácil configuración para diferentes casos de uso
5. **Compatibilidad**: Se integra con tu código existente sin cambios mayores

## Resultados Esperados

- **Fase 1** (epochs 1-15): El modelo aprende a corregir blur ligero
- **Fase 2** (epochs 15-25): Mejora en blur moderado
- **Fase 3** (epochs 25-40): Maneja blur fuerte sin emborronar
- **Fase 4** (epochs 40-50): Generalización a todos los niveles

## Troubleshooting

### Si el modelo sigue emborrando:

1. **Reduce el learning rate**: `--global_lr 1e-4`
2. **Aumenta weight decay**: `--weight_decay 0.05`
3. **Usa curriculum más gradual**: Entrena más epochs con dificultad fija
4. **Ajusta loss weights**: Aumenta el peso de SSIM: `--loss_weights 0.2 1.5 0.3`

### Si la convergencia es lenta:

1. **Aumenta batch size**: `--batch_size 8`
2. **Usa learning rate más alto**: `--global_lr 5e-4`
3. **Reduce dropout**: `--dropout_p 0.05`

## Monitoreo del Entrenamiento

En wandb, observa estas métricas:
- **train_ssim_loss**: Debe disminuir consistentemente
- **val_ssim_loss**: No debe aumentar (overfitting)
- **train_sobel_loss**: Preservación de bordes
- **curriculum_stage**: Progresión automática de dificultad

## Próximos Pasos

1. Entrena con curriculum learning por 50 epochs
2. Evalúa en tus imágenes de test con diferentes sigmas
3. Si los resultados son buenos, entrena por más epochs
4. Considera fine-tuning con dificultad específica para tu caso de uso

La clave es que ahora el modelo aprende gradualmente a manejar diferentes niveles de desenfoque, en lugar de intentar aprender todo a la vez, lo que causaba que emborrara las imágenes.