# BioApp

BioApp is a Windows desktop application for analyzing experimental data and preparing figures. Import a CSV, TSV, Excel workbook, or Prism XML table, fit a curve or compare groups, and export the resulting graph. A `.biofig` project keeps tables, analyses, graphs, and figure layouts together.

## Run on Windows (x64)

1. Open **Releases** in this repository. Under **BioApp 0.1.0 → Assets**, download `bioapp-0.1.0-windows-x64.zip`.
2. Right-click the ZIP, choose **Extract All**, and open the extracted `bioapp` folder.
3. Double-click `bioapp.exe`. Keep it alongside the `_internal` folder, which contains Python, Qt, and the analysis libraries.

The portable package runs without a separate Python installation, virtual environment, or activation key. The GitHub **Source code (zip)** download is for developers.

## Run from source on Windows

The repository contains the source for BioApp 0.1.0. Its Windows build uses 64-bit Python 3.14.6.

1. Install 64-bit Python 3.14 with the Windows `py` command.
2. On the GitHub repository page, choose **Code → Download ZIP** and extract the archive. Open PowerShell in the extracted folder that contains `requirements.lock.txt`. If you already cloned the repository, open PowerShell in its root folder.
3. Create a Python environment and install the pinned dependencies. Run these commands once from that folder.

   ```powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
   ```

4. Start BioApp. Use the same command from the same folder each time you want to open it.

   ```powershell
   .\.venv\Scripts\python.exe -m bioapp
   ```

## Build a portable Windows copy

After installing the dependencies, run the following command in the repository root.

```powershell
.\.venv\Scripts\python.exe -m PyInstaller packaging\bioapp.spec
```

The build creates `dist\bioapp\`. To use it on another 64-bit Windows computer, copy the entire `bioapp` folder, keeping `bioapp.exe` and `_internal` together. Double-click `bioapp.exe` to launch. The folder includes its own Python runtime.

## License

BioApp is distributed under the MIT License in `LICENSE`. The bundled runtime, libraries, and fonts retain their own licenses. The Windows package includes their notices in `licenses\`.
