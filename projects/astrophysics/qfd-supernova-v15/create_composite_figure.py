#!/usr/bin/env python3
"""
Create a composite figure showing all 4 generated publication figures.
Useful for quick review and presentations.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


def main():
    fig = plt.figure(figsize=(16, 14))

    # Define the 4 figures
    figures = [
        ("results/mock_stage3/figures/fig4_hubble_diagram.png", "Figure 4: Hubble Diagram"),
        ("results/mock_stage3/figures/fig5_corner_plot.png", "Figure 5: Posterior Corner Plot"),
        ("results/mock_stage3/figures/fig6_per_survey_residuals.png", "Figure 6: Per-Survey Residuals"),
        ("results/mock_stage3/figures/fig8_holdout_performance.png", "Figure 8: Hold-out Performance"),
    ]

    # Create 2x2 grid
    for i, (path, title) in enumerate(figures, 1):
        ax = fig.add_subplot(2, 2, i)

        # Load and display image
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis('off')

        # Add title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.suptitle("QFD V15 Pipeline: Publication Figures (Mock Data Demo)",
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save composite
    out_path = Path("results/mock_stage3/figures/composite_all_figures.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved composite figure: {out_path}")
    plt.close()

    print("\n✓ Composite figure generation complete!")
    print("\nGenerated figures:")
    for path, title in figures:
        print(f"  - {title}")


if __name__ == "__main__":
    main()
