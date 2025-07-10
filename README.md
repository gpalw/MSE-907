# MSE-907

---

## Quick Start

1. **Clone the repository:**
    ```bash
    git clone https://github.com/gpalw/MSE-907
    cd MSE-907
    ```
2. **Build and run all services locally (requires Docker and Docker Compose):**
    ```bash
    cd deploy
    docker-compose up --build
    ```
3. **Access the web interface:**
    - Go to `http://localhost` in your browser.

---

## Requirements

- Python 3.10+ & Java 17+
- Docker & Docker Compose
- AWS account (for cloud deployment)

**Key Python dependencies:**
- TensorFlow 2.x, pandas, numpy, FastAPI, Streamlit, etc.
**Key Java dependencies:**
- Spring Boot

---

## Usage

- **Upload Data:** Use the Streamlit web interface to upload your sales CSV file.
- **Forecast:** The system will automatically process your data and provide sales forecasts.
- **View Results:** Forecasts and related visualizations will appear in the dashboard.

---

## Development & Deployment

- **Local development:** See `dockerBuilder.sh` and scripts in `/deploy`.
- **Cloud deployment:** Recommended on AWS EC2 (t3.medium+). For deployment instructions, see comments in `dockerBuilder.sh` and the deployment section of this README.(Modify ACCOUNT_ID and REGION and make sure the system has AWS commands installed and logged in)
- **Model updates:** Place new pre-trained models in `/models` and update `model_registry.json` as needed.

---

## Directory Structure

- **data**  
  Original data sources and sample datasets used by the project.

- **deploy**  
  Deployment scripts and automation files (Docker, cloud deployment, etc.).

- **models**  
  Pre-trained model files and model registry configurations.

- **project**  
  Main project codebase, including all service implementations.
    - **forecasting-platform**: Backend platform service (Java/Spring Boot).
    - **python-service**: Machine learning inference service (FastAPI).
    - **streamlit-ui**: Streamlit-based web UI for data upload and results.
    - **build_push_*.sh**: Shell scripts to build and push Docker images for each service.

- **record**  
  Weekly progress reports and logs.

- **src**  
  Data cleaning and preprocessing utility scripts.

- **video**  
  Project demo videos, including deployment and feature demonstrations.

---

## Contributors

- Wen Liang ([gpalw](https://github.com/gpalw))
- Supervisor: Dr. Mukesh Mishra

---

## References

See the [References](./References) section in the project report for a full list of papers and online resources cited.

---

*This project is part of the MSE907 Industry-based Capstone Research Project, Yoobee College, 2025.*
