# Research scripts

Experiments around the PyTorch model in `emotionflow_llm/`. Nothing here is used by the web app.

Run from the repository root (each script adds the root to `sys.path` and writes figures to `docs/images/`):

```bash
pip install -r requirements-dev.txt
python research/visualize.py
python research/neuron_emotion_viz.py
python research/emotion_conditioning_demo.py
```

Note: the model is randomly initialised (no training pipeline yet), so figures illustrate the *mechanics*, not learned behaviour. See `docs/ROADMAP.md`.
