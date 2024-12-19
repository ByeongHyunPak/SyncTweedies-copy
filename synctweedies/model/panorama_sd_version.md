# Model Version Log

## 0.0.0

- **목적**: Depth condition 없이 text prompt로만 inference 하기
- **변경사항**: `panorama_model.py`의 ControlNet -> StableDiffusion
- **관련 파일**: `config/panorama_sd_config.py`

## 0.0.1

- **목적**: HIWYN의 (1) Conditional Upsampling, (2) Discrete Warping 으로 $w^{(0)}_j$ sampling