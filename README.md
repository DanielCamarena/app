# Data and Compute Reality Check

Interactive Streamlit app for an engineering AI assignment:
**"The Data & Compute Reality Check: Neural Networks/Transformers vs Physics-Informed Neural Networks (PINNs)"**

Physical system: **2D heat diffusion in a plate**.

## Project Goals

This app explains, with technical rigor and executive clarity:

- The physics of heat diffusion in a 2D plate.
- Differences between traditional Neural Networks / Transformers and PINNs.
- Trade-offs between both modeling families.
- Compute reality: hardware, memory, time, and cost constraints.
- Matrix operation bottlenecks and scaling intuition.
- Data quality bottlenecks and leakage risks.
- Final recommendation for engineering management decisions.

## Project Structure

- `app.py`: Main Streamlit application.
- `utils.py`: Synthetic data generation, compute proxy functions, and plotting helpers.
- `requirements.txt`: Minimal dependencies.
- `.gitignore`: Python/Streamlit-oriented ignore rules.

## Local Setup

### 1. Clone or open the project

```bash
git clone <YOUR_REPO_URL>
cd APP
```

### 2. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser (typically `http://localhost:8501`).

## Upload to GitHub

If this is your first commit:

```bash
git init
git add .
git commit -m "Initial Streamlit app for data vs PINN reality check"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

If repo already exists, just commit and push:

```bash
git add .
git commit -m "Update Streamlit app"
git push
```

## Deploy on Streamlit Community Cloud

1. Push this project to a public GitHub repository.
2. Go to Streamlit Community Cloud and sign in with GitHub.
3. Click **New app**.
4. Select your repository and branch (`main`).
5. Set main file path to `app.py`.
6. Click **Deploy**.

Notes:

- No external APIs or keys are required.
- No database is required.
- Data is generated synthetically within the app.

## Academic and Technical Notes

- The app uses conceptual compute proxies for clarity, not hardware-calibrated benchmark results.
- The heatmap is synthetic and intended for communication and intuition building.
- The leakage section highlights practical mistakes in spatiotemporal ML validation.

## License

For educational use in coursework.
