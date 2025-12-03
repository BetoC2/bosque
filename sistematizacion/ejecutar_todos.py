import subprocess
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

if os.path.exists("base_de_datos.xlsx"):
    os.remove("base_de_datos.xlsx")

for archivo in ["beto.py", "carlos.py", "fabri.py", "sofi.py"]:
    subprocess.run(["uv", "run", archivo])
