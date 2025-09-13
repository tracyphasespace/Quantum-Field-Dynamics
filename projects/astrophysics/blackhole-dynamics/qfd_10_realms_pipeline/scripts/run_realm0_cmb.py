#!/usr/bin/env python3
import os, json, yaml
from realms.realm0_cmb import CMBTargets, run

def main():
    root = os.path.join(os.path.dirname(__file__), '..')
    y = yaml.safe_load(open(os.path.join(root, 'qfd_params', 'defaults.yaml'), 'r', encoding='utf-8'))
    cmb_y = y.get('cmb_targets', {})
    pol = cmb_y.get('polarization', {})
    cfg = CMBTargets(
        T_CMB_K=cmb_y.get('T_CMB_K', 2.725),
        allow_vacuum_birefringence=pol.get('allow_vacuum_birefringence', False),
        allow_parity_violation=pol.get('allow_parity_violation', False),
    )
    res = run({}, cfg)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
