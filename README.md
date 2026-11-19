# BioApp

BioApp is a Windows desktop application for analyzing experimental data and preparing figures. Import a CSV, TSV, Excel workbook, or Prism XML table, fit a curve or compare groups, and export the resulting graph. A `.biofig` project keeps tables, analyses, graphs, and figure layouts together.

The local Model Context Protocol (MCP) server, `bioapp.mcp_server`, exposes 28 tools for working with the same `.biofig` projects as the desktop app. For Octet biolayer interferometry (BLI) data, the desktop and MCP workflows below take an instrument export through to a labeled sensorgram.

## Run on Windows (x64)

1. On Windows 10 or 11 (x64), install the latest **Microsoft Visual C++ v14 Redistributable (x64)** once. Download `vc_redist.x64.exe` from Microsoft's official address, `https://aka.ms/vc14/vc_redist.x64.exe`. Run it, select **Install** if offered, and restart Windows if prompted. If the current x64 runtime is already installed, continue to step 2.
2. Open **Releases** in this repository. Under **BioApp 0.1.0 → Assets**, download `bioapp-0.1.0-windows-x64.zip`.
3. Right-click the ZIP, choose **Extract All**, and open the extracted `bioapp` folder.
4. Double-click `bioapp.exe`. Keep it alongside the `_internal` folder, which contains Python, Qt, and the analysis libraries.

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

## Octet BLI sensorgrams

1. In Octet/ForteBio Data Analysis, export **Kinetic Data** as a CSV. Use one export per graph. Each sensor has a time column and a response column (for example, `Time1,Data1`), with an optional fitted response (`Fit1`). Keep the metadata rows above the column headings so BioApp can read concentrations and phase times.
2. In BioApp, choose **Import traces** (`Ctrl+Shift+I`) and select the file. BioApp plots response in nanometers against time in seconds, labels the raw curves by concentration when available, and overlays any fitted curves supplied by the instrument.
3. Check the transition from association to dissociation. BioApp uses the median `Start2` time when present. Otherwise, it estimates the boundary from turning points in the exported fits when available. You can adjust the boundary, set the association start to zero, and choose whether to show raw curves, fits, or both.
4. Add kinetic annotations if needed. Values such as `KD`, `kon`, and `koff` come from the export or from values you enter. This workflow displays those results without performing a new kinetic fit. Save the editable project as `.biofig` and export the graph as SVG, PDF, or a 600 dpi PNG.

## Build a portable Windows copy

After installing the dependencies, run the following command in the repository root.

```powershell
.\.venv\Scripts\python.exe -m PyInstaller packaging\bioapp.spec
```

The build creates `dist\bioapp\`. To use it on another 64-bit Windows computer, copy the entire `bioapp` folder, keeping `bioapp.exe` and `_internal` together. Double-click `bioapp.exe` to launch. The folder includes its own Python runtime.

## Use with AI agents (MCP)

The MCP server runs locally and communicates with its client over standard input and output (stdio). Each project edit is saved to disk. If the desktop app has the same project open, it reloads external changes and asks before replacing unsaved edits.

**Install once.** Follow steps 1–3 in “Run from source on Windows” to create `.venv` and install the pinned dependencies, including the `mcp` package. MCP clients launch the Python module from this source installation. The portable EXE launches the desktop interface.

**Register the server.** Replace `C:\path\to\bioapp` with the absolute path of the repository folder.

- Claude Code:

  ```powershell
  claude mcp add bioapp -s user -e "PYTHONPATH=C:\path\to\bioapp" -- "C:\path\to\bioapp\.venv\Scripts\python.exe" -X utf8 -m bioapp.mcp_server
  ```

- Codex (`~/.codex/config.toml`):

  ```toml
  [mcp_servers.bioapp]
  command = 'C:\path\to\bioapp\.venv\Scripts\python.exe'
  args = ['-X', 'utf8', '-m', 'bioapp.mcp_server']
  startup_timeout_sec = 60

  [mcp_servers.bioapp.env]
  PYTHONPATH = 'C:\path\to\bioapp'
  ```

- Other MCP clients (JSON `mcpServers` configuration):

  ```json
  {
    "mcpServers": {
      "bioapp": {
        "command": "C:\\path\\to\\bioapp\\.venv\\Scripts\\python.exe",
        "args": ["-X", "utf8", "-m", "bioapp.mcp_server"],
        "env": { "PYTHONPATH": "C:\\path\\to\\bioapp" }
      }
    }
  }
  ```

Restart the agent or its MCP connection afterwards. The server describes its own workflow to the agent, so no extra prompt is needed.

**What an agent can do.** A typical sequence:

1. Open or create a project: `project_open` / `project_new`.
2. Add data: `table_add` / `table_import` (CSV, TSV, Excel, Prism `.pzfx`).
3. Analyze: `analysis_run` for curve fits (`model_search` finds models), t tests, ANOVA, nonparametric, contingency, survival, regression, PCA and more.
4. Graph: `graph_create`, `graph_preview` (the agent looks at the rendered PNG), `graph_update` / `graph_look` for styling.
5. Export: `graph_export` / `layout_export` to SVG, PDF, PNG, TIFF, or EPS.

Use `trace_import` for chromatography UV/conductivity traces and Octet BLI sensorgrams. The import returns a `graph_id` for preview, formatting, and export. `project_undo` can restore up to 10 previous saved versions.

**Limit file access (recommended).** By default the server can read and write any path the agent names. Set `BIOAPP_MCP_ROOTS` to a `;`-separated list of folders, and every project, import, and export path must then lie inside one of them. For example, add `-e "BIOAPP_MCP_ROOTS=D:\data;D:\figures"` to the Claude Code command, or the same key under `env` in the other configurations.

### Example: export a BLI sensorgram over MCP

Save the example as `export_bli.py` in the repository root. From that folder, run `.\.venv\Scripts\python.exe export_bli.py "C:\data\Kinetic Data.csv" "C:\data\run01.biofig"`. Choose a new project filename in an existing folder. The script saves the project and writes an SVG with the same stem.

```python
import asyncio
import os
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters


async def call(client, tool, **args):
    result = await client.call_tool(tool, args)
    if result.is_error:
        raise RuntimeError(f"{tool}: {result.content[0].text}")
    return result.structured_content


async def main(csv, project):
    server = StdioServerParameters(
        command=sys.executable,
        args=["-X", "utf8", "-m", "bioapp.mcp_server"],
        cwd=str(Path(__file__).resolve().parent),
        env={"BIOAPP_MCP_ROOTS": os.environ.get("BIOAPP_MCP_ROOTS", "")},
    )
    async with Client(server) as client:
        await call(client, "project_new", path=str(project), title=project.stem)
        imported = await call(client, "trace_import", paths=[str(csv)])
        graph = imported["graph_id"]
        await call(client, "graph_look", graph_id=graph, name="bold")
        await call(client, "graph_update", graph_id=graph,
                   patch={"trace": {"bli": {"rebase": True, "phase_labels": True}}})
        await call(client, "graph_export", graph_id=graph, path=str(project.with_suffix(".svg")))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
```

## Planned instrument support

Future releases will add support for activity curves from different microplate readers, one instrument export format at a time.

## License

BioApp is distributed under the MIT License in `LICENSE`. The bundled runtime, libraries, and fonts retain their own licenses. The Windows package includes their notices in `licenses\`.
