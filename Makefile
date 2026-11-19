.DEFAULT_GOAL := replay-statistics
.PHONY: env env-vision data source-export source-reconcile source-analysis replay-statistics thesis-analysis thesis-extension thesis-diagnostics thesis-ordering thesis-figures thesis-figure-refresh thesis-recording-support thesis-weighted-audit thesis-model-fit thesis-timing-weights pipeline-smoke pipeline-v4-smoke test lint format all calibrated-speeds calibrated-speeds-replay robustness robustness-replay clean

PYTHON ?= python3.11
VENV ?= .venv
PROJECT_PYTHON = PYTHONPATH=$(CURDIR)/src $(PYTHON)
REPLAY_PANEL ?= results/monitor_quality_reanalysis_20261119/extension_hourly_panel.csv
REPLAY_OUTPUT ?= build/panel_replay
REPLAY_CORE_RESAMPLES ?= 20000
REPLAY_EXTENSION_RESAMPLES ?= 2000
REPLAY_EXPECTED_BUNDLE ?=
PREFLIGHT_REPORT ?= build/data_preflight.json
ANALYSIS_CONFIG ?= data/manifests/analysis-config-v4.json
ANALYSIS_OUTPUT ?= build/analysis_v4
V4_SMOKE_OUTPUT ?= build/v4_controlled_smoke
SPEED_DATABASE ?= data/results.db
SPEED_CONFIG ?= data/manifests/production-speed-v4.json
SPEED_OUTPUT ?= build/production_speed_v4
SPEED_REPLAY_OUTPUT ?= build/production_speed_v4_replay
SOURCE_DATABASE ?= data/results.db
SOURCE_OUTPUT ?= build/raw_source_export
SOURCE_BUNDLE ?= build/raw_source_export
SOURCE_RECONCILIATION_OUTPUT ?= build/source_reconciliation
REANALYSIS_PANEL ?= results/v4_all_weather/phase5_hourly_paired_dataset_all_weather.csv
REANALYSIS_OUTPUT ?= build/thesis_analysis
EXTENSION_BASE ?= results/recorded_hourly_reanalysis_20261109
EXTENSION_OUTPUT ?= build/thesis_extension
EXTENSION_CORE_RESAMPLES ?= 20000
EXTENSION_RESAMPLES ?= 2000
DIAGNOSTIC_PANEL ?= $(EXTENSION_BASE)/recorded_hourly_panel.csv
DIAGNOSTIC_OUTPUT ?= build/thesis_diagnostics
ORDERING_SOURCE ?= results/monitor_quality_reanalysis_20261119
ORDERING_OUTPUT ?= build/thesis_ordering
FIGURE_EXTENSION ?= results/monitor_quality_reanalysis_20261119
FIGURE_MEASUREMENT ?= results/measurement_model_extension_20261109
FIGURE_OUTPUT ?= build/thesis_additional_figures
FIGURE_ORDERING ?= results/monitor_quality_ordering_20261119
FIGURE_PROFILE ?= results/thesis_figure_refresh_20261117/fig_model_profile_response_checked.png
FIGURE_REFRESH_OUTPUT ?= build/thesis_figure_refresh
RECORDING_SUPPORT_OUTPUT ?= build/thesis_recording_support
WEIGHTED_AUDIT_OUTPUT ?= build/thesis_weighted_audit
MODEL_FIT_OUTPUT ?= build/thesis_model_fit
TIMING_WEIGHTS_OUTPUT ?= build/thesis_timing_weights
RECORDING_SUPPORT_BUNDLE ?= results/recording_support_sensitivity_20261117
ROBUSTNESS_CONFIG ?= data/manifests/robustness-config.json
ROBUSTNESS_OUTPUT ?= build/robustness_run

env:
	@if [ ! -x "$(VENV)/bin/python" ]; then $(PYTHON) -m venv $(VENV); fi
	$(VENV)/bin/python -m pip install --upgrade pip==25.0.1
	$(VENV)/bin/python -m pip install --requirement requirements-dev.lock
	$(VENV)/bin/python -m pip install --no-deps -e .
	$(VENV)/bin/python -c "import vehicle_emission"

env-vision: env
	$(VENV)/bin/python -m pip install -e .[vision]

data:
	$(PROJECT_PYTHON) -m vehicle_emission.data_preflight --report $(PREFLIGHT_REPORT)

source-export:
	$(PROJECT_PYTHON) -m vehicle_emission.source_export \
		--database "$(SOURCE_DATABASE)" --output-directory "$(SOURCE_OUTPUT)"

source-reconcile:
	$(PROJECT_PYTHON) -m vehicle_emission.source_reconciliation \
		--source-directory "$(SOURCE_BUNDLE)" --output-directory "$(SOURCE_RECONCILIATION_OUTPUT)"

thesis-analysis:
	$(PROJECT_PYTHON) -m vehicle_emission.inference.recorded_hourly_reanalysis \
		--panel "$(REANALYSIS_PANEL)" \
		--ambient "$(SOURCE_BUNDLE)/air_pollution_history.csv" \
		--weather "$(SOURCE_BUNDLE)/weather_history.csv" \
		--raw-source-directory "$(SOURCE_BUNDLE)" \
		--rainfall-mode dry_only --resamples 1000 --seed 42 \
		--output-directory "$(REANALYSIS_OUTPUT)"

thesis-extension:
	$(PROJECT_PYTHON) -m vehicle_emission.inference.recorded_hourly_extension \
		--base-bundle "$(EXTENSION_BASE)" --source-directory "$(SOURCE_BUNDLE)" \
		--core-resamples $(EXTENSION_CORE_RESAMPLES) --extension-resamples $(EXTENSION_RESAMPLES) \
		--seed 42 --output-directory "$(EXTENSION_OUTPUT)"

thesis-diagnostics:
	$(PROJECT_PYTHON) -m vehicle_emission.measurement.thesis_diagnostics \
		--repository-root "$(CURDIR)" --raw-source-directory "$(SOURCE_BUNDLE)" \
		--hourly-panel "$(DIAGNOSTIC_PANEL)" --output-directory "$(DIAGNOSTIC_OUTPUT)"

thesis-recording-support:
	$(PROJECT_PYTHON) -m vehicle_emission.inference.recording_support \
		--source-directory "$(SOURCE_BUNDLE)" --extension-bundle "$(ORDERING_SOURCE)" \
		--output-directory "$(RECORDING_SUPPORT_OUTPUT)" --resamples 2000 \
		--seed 42 --block-days 1 7

thesis-weighted-audit:
	$(PROJECT_PYTHON) -m vehicle_emission.measurement.weighted_count_audit \
		--extension-directory "$(FIGURE_EXTENSION)" \
		--measurement-directory "$(FIGURE_MEASUREMENT)" \
		--output-directory "$(WEIGHTED_AUDIT_OUTPUT)"

thesis-ordering:
	$(PROJECT_PYTHON) scripts/thesis_ordering_figures.py \
		--source-directory "$(ORDERING_SOURCE)" --output-directory "$(ORDERING_OUTPUT)"

thesis-model-fit:
	$(PROJECT_PYTHON) -m vehicle_emission.inference.model_fit_diagnostics \
		--output-directory "$(MODEL_FIT_OUTPUT)"

thesis-timing-weights:
	$(PROJECT_PYTHON) -m vehicle_emission.inference.timing_weight_sensitivity \
		--source-directory "$(SOURCE_BUNDLE)" --extension-bundle "$(FIGURE_EXTENSION)" \
		--recording-support-bundle "$(RECORDING_SUPPORT_BUNDLE)" \
		--output-directory "$(TIMING_WEIGHTS_OUTPUT)" --resamples 2000 --seed 42

thesis-figures:
	$(PROJECT_PYTHON) -m vehicle_emission.visualisation.thesis_figures \
		--extension-directory "$(FIGURE_EXTENSION)" \
		--measurement-directory "$(FIGURE_MEASUREMENT)" \
		--output-directory "$(FIGURE_OUTPUT)"

thesis-figure-refresh:
	$(PROJECT_PYTHON) -m scripts.refresh_thesis_figures \
		--extension-directory "$(FIGURE_EXTENSION)" \
		--measurement-directory "$(FIGURE_MEASUREMENT)" \
		--ordering-directory "$(FIGURE_ORDERING)" \
		--profile-image "$(FIGURE_PROFILE)" \
		--output-directory "$(FIGURE_REFRESH_OUTPUT)"

pipeline-smoke:
	$(PROJECT_PYTHON) -m vehicle_emission.reproduction smoke --output-dir build/pipeline_smoke

pipeline-v4-smoke:
	$(PROJECT_PYTHON) -m vehicle_emission.inference.v4_smoke --output-directory $(V4_SMOKE_OUTPUT)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m black --check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m black src tests

all: replay-statistics

replay-statistics:
	$(PROJECT_PYTHON) -m vehicle_emission.inference.panel_replay \
		--panel "$(REPLAY_PANEL)" --output-directory "$(REPLAY_OUTPUT)" \
		--core-resamples $(REPLAY_CORE_RESAMPLES) \
		--extension-resamples $(REPLAY_EXTENSION_RESAMPLES) --seed 42 \
		$(if $(REPLAY_EXPECTED_BUNDLE),--expected-bundle "$(REPLAY_EXPECTED_BUNDLE)")

source-analysis:
	$(PROJECT_PYTHON) -m vehicle_emission.inference.frozen_analysis execute-v4 \
		--config $(ANALYSIS_CONFIG) \
		--output-directory $(ANALYSIS_OUTPUT) \
		--repository-root $(CURDIR)

calibrated-speeds:
	$(PROJECT_PYTHON) -m vehicle_emission.calculate_speeds \
		--db-path $(SPEED_DATABASE) --config $(SPEED_CONFIG) --output-dir $(SPEED_OUTPUT)

calibrated-speeds-replay:
	$(PROJECT_PYTHON) -m vehicle_emission.calculate_speeds \
		--replay-bundle $(SPEED_OUTPUT) --output-dir $(SPEED_REPLAY_OUTPUT)

robustness:
	$(PROJECT_PYTHON) -m vehicle_emission.robustness.runner execute \
		--config $(ROBUSTNESS_CONFIG) --output-directory $(ROBUSTNESS_OUTPUT) \
		--repository-root $(CURDIR)

robustness-replay:
	$(PROJECT_PYTHON) -m vehicle_emission.robustness.runner replay \
		--output-directory $(ROBUSTNESS_OUTPUT) --repository-root $(CURDIR)

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
