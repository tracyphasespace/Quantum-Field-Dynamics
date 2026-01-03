# Raw Data Directory

## NUBASE2020 Data

This directory should contain the NUBASE2020 nuclear data file.

### Download Instructions

1. Visit the IAEA Atomic Mass Data Center:
   https://www-nds.iaea.org/amdc/

2. Download the NUBASE2020 file:
   - File: `nubase_1.mas20.txt` (or similar)
   - Format: Fixed-width text file
   - Contains: Nuclear masses, half-lives, decay modes

3. Place the file in this directory:
   ```
   data/raw/nubase_1.mas20.txt
   ```

### Alternative: Direct Download

```bash
cd data/raw/
wget https://www-nds.iaea.org/amdc/ame2020/nubase_1.mas20.txt
```

### File Format

The NUBASE format is a fixed-width text file with columns:
- Mass number (A)
- Atomic number (Z)
- Element symbol
- Mass excess (keV)
- Excitation energy (keV)
- Half-life
- Decay modes
- And more...

The parser (`src/parse_nubase.py`) handles this format automatically.

### Citation

If you use NUBASE2020 data, please cite:

```
Kondev, F. G., Wang, M., Huang, W. J., Naimi, S., & Audi, G. (2021).
The NUBASE2020 evaluation of nuclear physics properties.
Chinese Physics C, 45(3), 030001.
```

### File Size

The NUBASE file is approximately 1-2 MB (text format).
