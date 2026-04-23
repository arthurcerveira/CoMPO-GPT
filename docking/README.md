Prepare receptors/targets
```bash
python prepare_receptors.py
python generate_box_coodinates.py
```

For generated ligands from the baselines:
```bash
python prepare_ligands.py
python prepare_docking_script.py
```

For active and inactive ligands from ExCAPE-DB:
```bash
python prepare_targets_ligands.py
python prepare_docking_script_targets.py
```