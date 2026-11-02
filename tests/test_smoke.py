import os
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_constructs_headlessly():
    from PySide6.QtWidgets import QApplication

    import scd_app
    from scd_app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert scd_app is not None
    assert (
        window.windowTitle()
        == "SCD Studio - EMG Decomposition, Edition & Visualisation"
    )
    assert window.tabs.count() == 4

    window.close()
    app.processEvents()


def test_help_menu_exposes_support_diagnostics(tmp_path):
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    help_menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text() == "&Help"
    )

    assert help_menu is not None
    assert "Open &Log Folder" in [action.text() for action in help_menu.actions()]
    assert "&Copy Diagnostics" in [action.text() for action in help_menu.actions()]

    log_path = tmp_path / "scd-studio.log"
    with (
        patch("scd_app.gui.main_window.configure_logging", return_value=log_path),
        patch(
            "scd_app.gui.main_window.runtime_diagnostics",
            return_value="test diagnostics",
        ),
    ):
        window._copy_diagnostics()

    assert QApplication.clipboard().text() == "test diagnostics"

    window.close()
    app.processEvents()


def test_config_cards_reemit_parameterless_change_signal():
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.tabs.config_tab import AuxChannelCard, GridCard

    app = QApplication.instance() or QApplication([])

    grid = GridCard(1, "#4a9eff")
    grid_changes = []
    grid.changed.connect(lambda: grid_changes.append(True))
    grid.name_edit.setText("Biceps")
    grid.muscle_edit.setText("Biceps brachii")
    if grid.config_combo.count() > 1:
        grid.config_combo.setCurrentIndex(1)
    assert len(grid_changes) >= 2

    aux = AuxChannelCard(1)
    aux_changes = []
    aux.changed.connect(lambda: aux_changes.append(True))
    aux.name_edit.setText("Force")
    aux.type_combo.setCurrentIndex(1)
    aux.start_spin.setValue(1)
    aux.end_spin.setValue(2)
    aux.field_edit.setText("signal.force")
    aux.unit_edit.setText("N")
    aux.mvc_edit.setText("100")
    assert len(aux_changes) >= 7

    grid.close()
    aux.close()
    app.processEvents()


def test_recording_controls_present_one_clear_selection_workflow():
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.tabs.config_tab import ConfigTab

    app = QApplication.instance() or QApplication([])
    tab = ConfigTab()

    assert tab.choose_recording_btn.text() == "Choose recording…"
    assert tab.choose_batch_btn.text() == "Choose batch…"
    assert tab.inspect_arrays_btn.text() == "Inspect arrays…"
    assert not tab.inspect_arrays_btn.isEnabled()

    tab.close()
    app.processEvents()


def test_copy_decomposition_configuration_confirms_updated_grids():
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QLineEdit,
        QMessageBox,
    )

    from scd_app.gui.tabs.decomposition_tab import DecompositionTab

    app = QApplication.instance() or QApplication([])

    def parameter_widgets(value: str):
        notch_filter = QComboBox()
        notch_filter.addItems(["None", "50"])
        peel_off = QComboBox()
        peel_off.addItems(["False", "True"])
        muap_window = QDoubleSpinBox()
        return {
            "sil_threshold": QLineEdit(value),
            "extension_factor": QLineEdit(value),
            "highpass_hz": QLineEdit(value),
            "lowpass_hz": QLineEdit(value),
            "notch_filter": notch_filter,
            "notch_harmonics": QCheckBox(),
            "peel_off": peel_off,
            "muap_window_ms": muap_window,
        }

    tab = DecompositionTab()
    source = parameter_widgets("42")
    target = parameter_widgets("1")
    source["notch_filter"].setCurrentText("50")
    source["notch_harmonics"].setChecked(True)
    source["peel_off"].setCurrentText("True")
    source["muap_window_ms"].setValue(25)
    tab.param_widgets = {"Grid 1": source, "Grid 2": target}

    with patch.object(QMessageBox, "information") as information:
        tab._copy_params_to_all("Grid 1")

    assert target["sil_threshold"].text() == "42"
    assert target["notch_filter"].currentText() == "50"
    assert target["notch_harmonics"].isChecked()
    assert target["peel_off"].currentText() == "True"
    assert target["muap_window_ms"].value() == 25
    information.assert_called_once_with(
        tab,
        "Configuration Applied to All Grids",
        'The decomposition configuration from "Grid 1" was copied to 1 other grid.',
    )

    tab.close()
    app.processEvents()


def test_decomposition_plot_shows_squared_source_and_markers():
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.tabs.decomposition_tab import DecompositionTab

    app = QApplication.instance() or QApplication([])
    tab = DecompositionTab()

    tab._plot_source_realtime(
        np.array([-2.0, 0.0, 3.0]),
        np.array([0, 2]),
        iteration=4,
        silhouette=0.95,
    )

    source_line, marker_line = tab.figure.axes[0].lines
    np.testing.assert_array_equal(source_line.get_ydata(), [4.0, 0.0, 9.0])
    np.testing.assert_array_equal(marker_line.get_ydata(), [4.0, 9.0])
    assert tab.figure.axes[0].get_ylabel() == "Squared amplitude"

    tab.close()
    app.processEvents()


def _force_overlay_tab():
    """DecompositionTab with a force channel whose unit was left blank."""
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import torch

    from scd_app.gui.tabs.decomposition_tab import DecompositionTab

    tab = DecompositionTab()
    tab.emg_data = torch.from_numpy(
        np.random.default_rng(0).standard_normal((2000, 4)).astype(np.float32)
    )
    tab.emg_path = Path("recording.h5")
    tab.grid_configs = {"Grid 1": {"channels": [0, 1, 2, 3], "params": {}}}
    aux = {
        "name": "Force",
        "unit": "",
        "source": "signal",
        "start_chan": 0,
        "end_chan": 1,
    }
    tab.config = SimpleNamespace(aux_channels=[aux], sampling_frequency=2048)
    return tab


def test_applying_config_tears_down_open_channel_rejection_view():
    """Re-applying the config used to leave the "Show force" CheckButtons hooked up."""
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    tab = _force_overlay_tab()
    config = tab.config

    errors, legend_labels, loops, timed_out = [], [], [], []
    tab.canvas.callbacks.exception_handler = errors.append

    class TrackedLoop(QEventLoop):
        def __init__(self):
            super().__init__()
            loops.append(self)

    def apply_config_mid_rejection():
        legend = tab.figure.axes[0].get_legend()
        if legend is not None:
            legend_labels.extend(t.get_text() for t in legend.get_texts())
        tab.setup_session(config, [tab.emg_path])
        tab.canvas.draw()

    def watchdog():
        timed_out.append(True)
        for loop in loops:
            loop.quit()

    QTimer.singleShot(0, apply_config_mid_rejection)
    QTimer.singleShot(5000, watchdog)
    with (
        patch("scd_app.gui.tabs.decomposition_tab.QEventLoop", TrackedLoop),
        patch.object(tab, "_load_emg_data", return_value=True),
        patch.object(tab, "_load_grid_configs"),
    ):
        tab._manual_channel_rejection()

    assert legend_labels == ["Force"]
    assert not timed_out, "channel-rejection loop was not exited"
    assert tab._setup_cancelled
    assert errors == []

    tab.close()
    app.processEvents()


@pytest.mark.parametrize("ending", ["confirm", "loop exited"])
def test_channel_rejection_continues_only_after_confirm(ending):
    """Closing the app used to leave the rejection view as if confirmed."""
    from matplotlib.backend_bases import MouseEvent
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    tab = _force_overlay_tab()
    errors, loops = [], []
    tab.canvas.callbacks.exception_handler = errors.append

    class TrackedLoop(QEventLoop):
        def __init__(self):
            super().__init__()
            loops.append(self)

    def end_view():
        if ending == "confirm":
            ax = next(
                a
                for a in tab.figure.axes
                if any(t.get_text() == "CONFIRM ALL" for t in a.texts)
            )
            x, y = ax.bbox.x0 + ax.bbox.width / 2, ax.bbox.y0 + ax.bbox.height / 2
            for name in ("button_press_event", "button_release_event"):
                MouseEvent(name, tab.canvas, x, y, button=1)._process()
        else:
            loops[-1].quit()  # what quitting the app does to the nested loop

    QTimer.singleShot(0, end_view)
    QTimer.singleShot(5000, lambda: [loop.quit() for loop in loops])  # watchdog
    with patch("scd_app.gui.tabs.decomposition_tab.QEventLoop", TrackedLoop):
        tab._manual_channel_rejection()
    # The caller's next step redraws the canvas
    tab.figure.clf()
    tab.canvas.draw()

    assert tab._setup_cancelled == (ending == "loop exited")
    assert errors == []

    tab.close()
    app.processEvents()


def test_applying_config_resets_pending_time_window_step(tmp_path):
    """Re-applying the config used to keep a pending "Confirm & Run" setup."""
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import torch
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.tabs import decomposition_tab
    from scd_app.gui.tabs.decomposition_tab import DecompositionTab

    app = QApplication.instance() or QApplication([])
    emg = torch.randn(4000, 16)

    def make_config(n_channels):
        electrode = SimpleNamespace(
            channels=list(range(n_channels)), type="surface", name="Test"
        )
        port = SimpleNamespace(enabled=True, name="Grid 1", electrode=electrode)
        return SimpleNamespace(
            ports=[port],
            aux_channels=[],
            sampling_frequency=2048,
            output_dir=str(tmp_path),
            data_layout=None,
        )

    def load_emg(tab):
        tab.emg_data = emg
        return True

    def reject_nothing(tab):
        tab.rejected_channels = [
            np.zeros(len(grid["channels"]), dtype=int)
            for grid in tab.grid_configs.values()
        ]

    with (
        patch.object(DecompositionTab, "_load_emg_data", load_emg),
        patch.object(DecompositionTab, "_load_emg_full", return_value=emg),
        patch.object(DecompositionTab, "_manual_channel_rejection", reject_nothing),
        patch.object(decomposition_tab, "DecompositionWorker") as worker,
    ):
        tab = DecompositionTab()
        tab.setup_session(make_config(8), [Path("session_a.h5")])
        tab.time_mode.setCurrentText("Manual selection")
        tab.start_btn.click()
        assert tab._awaiting_time_confirmation

        tab.setup_session(make_config(16), [Path("session_b.h5")])

        assert tab.start_btn.text() == "Start Decomposition"
        assert not tab._awaiting_time_confirmation
        assert tab.time_sel_widget.isHidden()
        assert tab.cancel_setup_btn.isHidden()
        assert tab.time_mode.isEnabled()

        tab.start_btn.click()  # Start: fresh setup for session B
        tab.start_btn.click()  # Confirm & Run
        _, grid_configs, rejected = worker.call_args.args[:3]
        assert worker.call_args.kwargs["emg_file_path"] == Path("session_b.h5")
        assert [len(m) for m in rejected] == [
            len(grid["channels"]) for grid in grid_configs.values()
        ]

    tab.close()
    app.processEvents()


def test_applying_config_is_refused_while_decomposition_runs():
    """Apply must not rebuild the grid configs under a running worker."""
    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication, QMessageBox

    from scd_app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = SimpleNamespace(sampling_frequency=2048, aux_channels=[], ports=[])

    with (
        patch.object(window.decomp_tab, "has_running_worker", return_value=True),
        patch.object(window.decomp_tab, "setup_session") as setup_session,
        patch.object(QMessageBox, "warning") as warning,
    ):
        window._on_config_applied(config, ["recording.h5"])

    setup_session.assert_not_called()
    warning.assert_called_once()
    assert window.config is None

    window.close()
    app.processEvents()
