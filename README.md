# Scene Cut Detection Using Histogram Distances

This project compares two methods for detecting scene cuts in videos using frame-to-frame histogram differences:

- **PDF (simple histogram distance)** – sensitive to noise, produces many false spikes  
- **CDF (cumulative histogram distance)** – stable, smooth, and produces a clear single spike at the true cut  

Across all tested videos, the **CDF method reliably detects the correct scene change**, while the PDF method does not.

## Method Overview
For each pair of consecutive frames:
1. Convert to grayscale  
2. Compute 256-bin histogram  
3. PDF: compute L1 distance between histograms  
4. CDF: compute cumulative histogram and then L1 distance  
5. Plot distance vs. frame index and detect the largest peak  

## Results
- **Video 1:** CDF detects cut at ~99 (PDF noisy)  
- **Video 2:** CDF detects cut at ~149  
- **Video 3:** CDF detects cut at ~174 (PDF produces false early peaks)

Plots in `/experiments/` show PDF vs CDF for each video.

## Usage
```bash
pip install numpy matplotlib mediapy
python main.py
