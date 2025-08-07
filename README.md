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