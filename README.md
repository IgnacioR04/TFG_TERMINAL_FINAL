# Terminal TFG

Terminal financiera estatica que sirve de dashboard visual del TFG
"Prediccion selectiva de BTC mediante ML", sin exponer el modelo.

## Arquitectura

```
  terminal_tfg.py  ─┐
                    ├──(genera)──>  index.html  ──(GitHub Pages)──>  web publica
  template.html   ─┘
```

- `terminal_tfg.py` descarga datos (yfinance + alternative.me), calcula
  indicadores tecnicos, un proxy de regimen (HMM light) y un semaforo
  de volatilidad, y los inyecta en el template.
- `template.html` es la UI estatica (Plotly + JS vanilla, estilo Revolut).
- `.github/workflows/update.yml` ejecuta el script cada 15 min via cron.

El modelo real del TFG (HMM + GARCH + XGBoost) **NO** forma parte de este
repositorio ni de los datos publicados. Lo que se muestra es un proxy
publico con los mismos principios de filtrado (regimen estable + vol baja).

## Despliegue

1. Clonar este repo en tu cuenta de GitHub.
2. Activar GitHub Pages: Settings > Pages > Source: `main` / `/ (root)`.
3. Activar Actions: Settings > Actions > Allow all actions.
4. Dar permisos de escritura al workflow: Settings > Actions > General >
   Workflow permissions > Read and write permissions.
5. El primer build se dispara al hacer push. Luego cada 15 minutos.

## Estructura de archivos

```
.
├── terminal_tfg.py           # Pipeline de datos
├── template.html             # Template de UI
├── index.html                # Generado (no tocar)
└── .github/
    └── workflows/
        └── update.yml        # Cron
```

## Uso local

```bash
pip install yfinance pandas numpy requests
python terminal_tfg.py
# Abrir index.html en el navegador
```

## Privacidad del modelo

La terminal distingue claramente entre:

- **Publico**: precio, indicadores tecnicos, regimen proxy, semaforo vol.
- **Privado**: predicciones XGBoost, probabilidades, trades del bot.

Las "ventanas activas" que aparecen en la grafica son un proxy visual
(regimen estable + vol bajo P80), no las senales reales del modelo.
