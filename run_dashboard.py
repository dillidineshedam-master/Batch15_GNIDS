import os
import sys

if __name__ == "__main__":
    dashboard_path = os.path.join("dashboard", "app.py")
    os.system(f"streamlit run {dashboard_path}")
