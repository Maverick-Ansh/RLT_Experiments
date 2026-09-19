"""Run this AFTER the main sweep (both run_sweep.py processes) has finished.

Does everything left: length generalization, cost bench, RL replay checks, depth-16 series,
figures, REPORT.md, then pushes the whole repo to a new public GitHub repo using the
Colab-Secrets GITHUB_TOKEN (Tools > Secrets in the Colab UI, notebook access enabled).

Safe to re-run: every step skips work that's already on disk.
"""
import os, subprocess, sys, time
sys.path.insert(0, '/content/RLT_Experiments')
os.chdir('/content/RLT_Experiments')

def sh(cmd, **kw):
    print('>>>', cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print('!!! nonzero exit:', r.returncode)
        print(r.stderr[-4000:])
    return r

t0 = time.time()

# 1. length generalization from the best checkpoints of the main sweep
sh('python scripts/length_gen.py --gpu 0 --tasks parity,s5_swaps,s5_standard')
sh('python scripts/length_gen.py --gpu 1 --tasks mod5_brackets,mod5_flat,addition')

# 2. RLT-2 chunk sweep (App. I.4), parity + s5_swaps, the two tasks where state matters most
for B in [1, 2, 4, 8, 16, 32]:
    sh(f'python scripts/run_sweep.py --gpu 0 --tasks parity,s5_swaps --archs rlt1_4+4 '
       f'--seeds 42,43,44 --steps 500 --lr 4e-4 --tag rlt2 --chunk {B}')

# 3. cost benchmark, RL replay, sixteen-layer series
sh('python scripts/cost_bench.py')
sh('python scripts/rl_replay.py')
sh('python scripts/depth16.py --gpu 0')

# 4. figures + report
sh('python scripts/aggregate.py')
sh('python scripts/lengthgen_figs.py')
sh('python scripts/rlt2_figs.py')
sh('python scripts/make_report.py')

print(f'\n--- finalize took {(time.time()-t0)/60:.1f} min, now pushing to GitHub ---\n')

# 5. push to GitHub using the Colab Secrets token (requires notebook access granted in the UI)
try:
    from google.colab import userdata
    token = userdata.get('GITHUB_TOKEN')
except Exception as e:
    token = None
    print('Could not read GITHUB_TOKEN from Colab Secrets:', e)
    print('Add it via the key icon in the left sidebar, grant this notebook access, and re-run '
          'just this cell.')

if token:
    user = sh('curl -s -H "Authorization: token $TOK" https://api.github.com/user',
              env={**os.environ, 'TOK': token})
    import json
    login = json.loads(user.stdout).get('login', 'Maverick-Ansh')
    repo = 'RLT_Experiments'
    sh(f'curl -s -X POST -H "Authorization: token $TOK" '
       f'https://api.github.com/user/repos -d \'{{"name":"{repo}","private":false}}\'',
       env={**os.environ, 'TOK': token})
    remote = f'https://{login}:{token}@github.com/{login}/{repo}.git'
    sh('git add -A')
    sh('git commit -m "RLT-1 from-scratch reproduction: architecture, 126-run sweep, '
       'length generalization, RLT-2 cost bench, RL replay checks\n\n'
       'Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"')
    sh(f'git remote remove origin 2>/dev/null; git remote add origin {remote.replace(token, "***")}')
    # set the real remote without printing the token
    subprocess.run(['git', 'remote', 'set-url', 'origin', remote], cwd='.')
    sh('git branch -M main')
    sh('git push -u origin main --force')
    print(f'\nPushed to https://github.com/{login}/{repo}')
else:
    print('\nSkipped push. Results, figures, and REPORT.md are all on disk under '
          '/content/RLT_Experiments — download the folder or push manually:')
    print('  cd /content/RLT_Experiments')
    print('  git add -A && git commit -m "RLT-1 reproduction"')
    print('  git remote add origin https://github.com/<you>/RLT_Experiments.git')
    print('  git push -u origin main')

print(f'\nTOTAL finalize+push time: {(time.time()-t0)/60:.1f} min')
