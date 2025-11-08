# Makefile for MNRAS Publication Figures
# QFD Supernova V15 Project

# Figure targets
FIGS = figure_hubble.pdf \
       figure_basis_inference.pdf \
       figure_residuals.pdf \
       figure_corner.pdf

# Lock files for reproducibility
LOCKS = environment.lock data.lock

# Default target: build all figures
.PHONY: all
all: $(FIGS)

# Build individual figures
figure_hubble.pdf: make_hubble.py mnras_style.py $(LOCKS)
	@echo "=== Building Figure 1: Hubble Diagram ==="
	python make_hubble.py
	@echo

figure_basis_inference.pdf: make_basis_inference.py mnras_style.py $(LOCKS)
	@echo "=== Building Figure 2: Basis & Inference ==="
	python make_basis_inference.py
	@echo

figure_residuals.pdf: make_residuals.py mnras_style.py $(LOCKS)
	@echo "=== Building Figure 3: Residual Diagnostics ==="
	python make_residuals.py
	@echo

figure_corner.pdf: make_corner.py mnras_style.py $(LOCKS)
	@echo "=== Building Figure 4: Corner Plot ==="
	python make_corner.py
	@echo

# Generate lock files
environment.lock: make_locks.py
	@echo "=== Generating environment lock ==="
	python make_locks.py --env
	@echo

data.lock: make_locks.py
	@echo "=== Generating data lock ==="
	python make_locks.py --data
	@echo

# Clean generated files
.PHONY: clean
clean:
	@echo "=== Cleaning generated files ==="
	rm -f $(FIGS)
	rm -f *_provenance.json
	@echo "Cleaned: figures and provenance files"

# Clean lock files (use with caution)
.PHONY: clean-locks
clean-locks:
	@echo "=== Cleaning lock files ==="
	rm -f $(LOCKS)
	@echo "Cleaned: lock files"

# Deep clean (figures + locks)
.PHONY: distclean
distclean: clean clean-locks
	@echo "=== Deep clean complete ==="

# Show status
.PHONY: status
status:
	@echo "=== Figure Generation Status ==="
	@echo "Figures:"
	@for fig in $(FIGS); do \
		if [ -f $$fig ]; then \
			echo "  ✓ $$fig ($(shell stat -c%s $$fig 2>/dev/null || stat -f%z $$fig) bytes)"; \
		else \
			echo "  ✗ $$fig (not built)"; \
		fi; \
	done
	@echo
	@echo "Lock files:"
	@for lock in $(LOCKS); do \
		if [ -f $$lock ]; then \
			echo "  ✓ $$lock"; \
		else \
			echo "  ✗ $$lock (not generated)"; \
		fi; \
	done
	@echo
	@echo "Scripts:"
	@for script in make_*.py; do \
		if [ -f $$script ]; then \
			echo "  ✓ $$script"; \
		fi; \
	done

# Validate figures (check that they're proper PDFs)
.PHONY: validate
validate: $(FIGS)
	@echo "=== Validating generated figures ==="
	@for fig in $(FIGS); do \
		if file $$fig | grep -q "PDF"; then \
			echo "  ✓ $$fig is a valid PDF"; \
		else \
			echo "  ✗ $$fig is NOT a valid PDF"; \
			exit 1; \
		fi; \
	done
	@echo "All figures validated successfully"

# Help target
.PHONY: help
help:
	@echo "QFD Supernova V15 - MNRAS Figure Generation"
	@echo
	@echo "Targets:"
	@echo "  make all          - Build all figures (default)"
	@echo "  make clean        - Remove generated figures"
	@echo "  make clean-locks  - Remove lock files"
	@echo "  make distclean    - Remove everything (figures + locks)"
	@echo "  make status       - Show build status"
	@echo "  make validate     - Validate generated PDFs"
	@echo "  make help         - Show this help"
	@echo
	@echo "Individual figures:"
	@echo "  make figure_hubble.pdf          - Figure 1: Hubble diagram"
	@echo "  make figure_basis_inference.pdf - Figure 2: Basis functions"
	@echo "  make figure_residuals.pdf       - Figure 3: Residual diagnostics"
	@echo "  make figure_corner.pdf          - Figure 4: Corner plot"
	@echo
	@echo "Lock files:"
	@echo "  make environment.lock - Generate Python environment lock"
	@echo "  make data.lock        - Generate data hash lock"
