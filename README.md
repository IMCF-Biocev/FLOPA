# FLOPA

**FLOPA - FLIM Open, Process, and Analyze**

A python based tool for FLIM data opening, processing and analysis. Planned as Napari plugin/widget.

Planned functionalities:
  - [ ] multi-format data import (.ptu, .sdt, and more)
  - [ ] header / metadata correction option
  - [ ] support for various acquisition parameters
  - [ ] preprocessing tools (filters, binning, noise reduction)
  - [ ] intensity and lifetime mapping
  - [ ] lifetime histograms
  - [ ] segmentation/ROI-based analysis
  - [ ] pixel/object-based phasor generation,
  - [ ] decay fitting


conda remove -n flopa --all

conda create -y -n flopa -c conda-forge python=3.10
conda activate flopa
python -m pip install "napari[all]"
python -m pip install "napari[all]" --upgrade
python -m pip install xarray
python -m pip install -U matplotlib
pip install -e .


```
FLOPA/
├── main.py                     # <-- to be made
└── flopa/
    ├── __init__.py
    ├── io/
    │   ├── __init__.py
    │   ├── decoder.py
    │   ├── file.py
    │   ├── reader.py
    │   └── reconstructor.py    # <-- MODIFIED
    ├── processing/
    │   ├── __init__.py
    │   ├── flim_image.py
    │   ├── phasor.py
    │   └── reconstruction.py   # <-- NEW
    └── widgets/
        ├── __init__.py
        ├── decay_panel.py      # <-- NEW
        ├── flim_widget.py      # <-- MODIFIED
        ├── phasor_panel.py
        ├── ptu_processing_panel.py # <-- MODIFIED
        └── utils/
            ├── __init__.py
            └── style.py        # <-- Placeholder
```