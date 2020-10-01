# Serving server

## Requirements

- Docker

## Start server

### CPU

```bash
sh ./start_inference_server.sh model_weights.pth model_code.py
```

### GPU

```bash
sh ./start_inference_server.sh model_weights.pth model_code.py $CUDA_VISIBLE_DEVICES
```

__note__: You might experience unexpected GPU allocation. By default CUDA devices are ordered by compute power
here devices are ordered by their PCI ID