"""
HSM Simulator TUI - Main Application
====================================
Textual-based TUI for the HSM Simulator.
"""

from __future__ import annotations
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Button, DataTable, Input, ListView, ListItem
from textual.screen import Screen, ModalScreen
from textual import work
from rich.console import Console
from rich.table import Table

import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.models import SimulationConfig, SweepConfig
from db.database import ConfigDB, SweepDB, init_db


class MainMenu(Screen):
    """Main menu screen."""
    
    CSS = """
    MainMenu {
        align: center middle;
    }
    MainMenu > Button {
        width: 40;
        margin: 0 0;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("\n╔══════════════════════════════════════════════════════════╗\n║        HSM Simulator v2.0 - Hierarchical Storage        ║\n║              DMF 7 + ESS 3500 + TS1160                 ║\n╚══════════════════════════════════════════════════════════╝\n", id="title")
        
        yield Button("📂 Parse DMF Log File", id="parse", variant="primary")
        yield Button("⚙️ Configure Simulation", id="config")
        yield Button("🔬 Run Parameter Sweep", id="sweep")
        yield Button("▶️ Run Single Simulation", id="run")
        yield Button("📊 Generate Report", id="report")
        yield Button("✅ Validate Against DMF", id="validate")
        yield Button("💾 Load/Save Configuration", id="loadsave")
        yield Button("🚪 Exit", id="exit", variant="error")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        button_id = event.button.id
        
        if button_id == "parse":
            self.app.push_screen(ParseScreen())
        elif button_id == "config":
            self.app.push_screen(ConfigScreen())
        elif button_id == "sweep":
            self.app.push_screen(SweepScreen())
        elif button_id == "run":
            self.app.push_screen(RunScreen())
        elif button_id == "report":
            self.app.push_screen(ReportScreen())
        elif button_id == "validate":
            self.app.push_screen(ValidateScreen())
        elif button_id == "loadsave":
            self.app.push_screen(LoadSaveScreen())
        elif button_id == "exit":
            self.app.exit()


class ParseScreen(Screen):
    """Parse DMF log file screen."""
    
    CSS = """
    ParseScreen {
        align: center middle;
    }
    
    #parse_container {
        width: 70;
        height: auto;
        border: solid blue;
        padding: 1 2;
    }
    
    Input {
        width: 100%;
        margin: 1 0;
    }
    
    #status {
        margin-top: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("📂 Parse DMF Log File", id="header"),
            Container(
                Static("Input Log File:"),
                Input(placeholder="/path/to/dmf.log", id="log_input"),
                Static("Output Trace File (optional):"),
                Input(placeholder="trace.jsonl", id="output_input"),
                Button("Parse", id="parse_btn", variant="primary"),
                Button("Browse...", id="browse_btn"),
                Button("Back", id="back_btn", variant="error"),
                id="buttons"
            ),
            Static("", id="status"),
            id="parse_container"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "parse_btn":
            self.parse_log()
    
    def parse_log(self):
        """Parse the DMF log file."""
        log_input = self.query_one("#log_input", Input)
        output_input = self.query_one("#output_input", Input)
        status = self.query_one("#status", Static)
        
        if not log_input.value:
            status.update("[red]Please enter a log file path[/red]")
            return
        
        log_path = Path(log_input.value)
        if not log_path.exists():
            status.update(f"[red]File not found: {log_path}[/red]")
            return
        
        status.update("[yellow]Parsing log file...[/yellow]")
        
        try:
            # Import and run parser
            from workload.dmf_log_parser import DMFLogParser
            
            parser = DMFLogParser(str(log_path))
            events = parser.parse_log()
            stats = parser.get_statistics()
            
            # Export if output specified
            if output_input.value:
                parser.export_normalized(output_input.value)
                status.update(
                    f"[green]Parsed {stats['total_events']} events\\n"
                    f"Recalls: {stats['recall_count']}, Migrations: {stats['migrate_count']}\\n"
                    f"Exported to: {output_input.value}[/green]"
                )
            else:
                status.update(
                    f"[green]Parsed {stats['total_events']} events\\n"
                    f"Recalls: {stats['recall_count']}, Migrations: {stats['migrate_count']}\\n"
                    f"Mean latency: {stats['latency_mean']:.2f}s[/green]"
                )
        except Exception as e:
            status.update(f"[red]Error: {str(e)}[/red]")


class ConfigScreen(Screen):
    """Configure simulation parameters."""
    
    CSS = """
    ConfigScreen {
        align: center middle;
    }
    
    #config_container {
        width: 80;
        height: auto;
        border: solid yellow;
        padding: 1 2;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("⚙️ Configure Simulation", id="header"),
            Static("ESS 3500 Configuration:", id="ess_header"),
            Static(
                "NSD Nodes: [4]  |  Throughput: [65] GiB/s  |  Capacity: [809] TiB  |  Network: [200Gb HDR IB]",
                id="ess_summary"
            ),
            Static("DMF Policy:", id="policy_header"),
            Static(
                "Age Threshold: [30] min  |  HWM: [80%]  |  LWM: [70%]  |  Large File: [100] GB",
                id="policy_summary"
            ),
            Static("TS1160 Configuration:", id="tape_header"),
            Static(
                "Drives: [16]  |  Rate: [400] MB/s  |  Mount Time: [~20]s",
                id="tape_summary"
            ),
            Static("Policy Preset:", id="preset_header"),
            Static("[balanced] - (maximum_performance | balanced | archive_optimized)", id="preset_summary"),
            Container(
                Button("Edit ESS 3500", id="edit_ess", variant="primary"),
                Button("Edit Policy", id="edit_policy", variant="primary"),
                Button("Edit TS1160", id="edit_tape", variant="primary"),
                Button("Select Preset", id="edit_preset", variant="primary"),
                id="edit_buttons"
            ),
            Container(
                Button("Save Configuration", id="save_btn", variant="success"),
                Button("Back", id="back_btn", variant="error"),
            ),
            id="config_container"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "save_btn":
            self.save_config()
    
    def save_config(self):
        """Save configuration to database."""
        # Placeholder - would save current config
        self.notify("Configuration saved!", severity="information")


class SweepScreen(Screen):
    """Parameter sweep configuration."""
    
    CSS = """
    SweepScreen {
        align: center middle;
    }
    
    #sweep_container {
        width: 80;
        height: auto;
        border: solid magenta;
        padding: 1 2;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("🔬 Parameter Sweep Configuration", id="header"),
            Static("Select Parameters to Sweep:", id="param_header"),
            Static("[ ] Age Threshold (minutes): 5, 10, 20, 30, 40, 50, 60, 120, 240, 480, 1440", id="age_info"),
            Static("[ ] HWM (%): 70, 75, 80, 85, 90, 95", id="hwm_info"),
            Static("[ ] Cache Priority: 0.1, 0.3, 0.5, 0.7, 1.0", id="priority_info"),
            Static("Replications per config: [30]", id="repl_info"),
            Container(
                Button("Configure Age Sweep", id="cfg_age", variant="primary"),
                Button("Configure HWM Sweep", id="cfg_hwm", variant="primary"),
                Button("Configure Priority Sweep", id="cfg_priority", variant="primary"),
                id="cfg_buttons"
            ),
            Container(
                Button("Start Sweep", id="start_btn", variant="success"),
                Button("Back", id="back_btn", variant="error"),
            ),
            id="sweep_container"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "start_btn":
            self.notify("Starting parameter sweep...", severity="information")


class RunScreen(Screen):
    """Run single simulation."""
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("▶️ Run Single Simulation", id="header"),
            Static("Ready to run with current configuration.", id="status"),
            Container(
                Button("Run", id="run_btn", variant="success"),
                Button("Back", id="back_btn", variant="error"),
            ),
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "run_btn":
            self.notify("Running simulation...", severity="information")


class ReportScreen(Screen):
    """Generate reports."""
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("📊 Generate Report", id="header"),
            Static("Available Sweeps:", id="sweeps_header"),
            Static("- No sweeps found. Run a parameter sweep first.", id="sweeps_list"),
            Container(
                Button("Generate CSV", id="csv_btn", variant="primary"),
                Button("Generate Charts", id="chart_btn", variant="primary"),
                Button("Back", id="back_btn", variant="error"),
            ),
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()


class ValidateScreen(Screen):
    """Validate against DMF."""
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("✅ Validate Against DMF", id="header"),
            Static("KS-Test Validation:", id="ks_header"),
            Static("Not yet configured. Run a simulation first.", id="status"),
            Container(
                Button("Run KS-Test", id="ks_btn", variant="primary"),
                Button("Verify Little's Law", id="ll_btn", variant="primary"),
                Button("Back", id="back_btn", variant="error"),
            ),
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()


class LoadSaveScreen(Screen):
    """Load/save configurations."""
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static("💾 Load/Save Configuration", id="header"),
            Static("Saved Configurations:", id="configs_header"),
            Static("- default (current)", id="configs_list"),
            Container(
                Button("Save Current", id="save_btn", variant="success"),
                Button("Load", id="load_btn", variant="primary"),
                Button("Delete", id="delete_btn", variant="error"),
                Button("Back", id="back_btn", variant="error"),
            ),
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back_btn":
            self.app.pop_screen()


class HSMSimulatorApp(App):
    """HSM Simulator Textual Application."""
    
    TITLE = "HSM Simulator v2.0"
    SUB_TITLE = "Hierarchical Storage Management"
    
    CSS_PATH = None
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "pop_screen", "Back"),
    ]
    
    def __init__(self):
        super().__init__()
        # Initialize database
        init_db()
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield MainMenu()
        yield Footer()


def main():
    """Main entry point."""
    app = HSMSimulatorApp()
    app.run()


if __name__ == "__main__":
    main()
