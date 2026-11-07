"""index.src.html + train/data.json + train/probes.json -> index.html (one file, with the data
inlined; three.js still loads from cdn.jsdelivr.net, and the page says so quietly if it does not).
Numbers in the static text are filled from train/summary.json, never typed by hand. The few
sentences written by hand about the results are checked against summary.json below, so a
retrain that changes them stops the build instead of shipping stale prose."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'index.src.html')).read()
data = json.load(open(os.path.join(HERE, 'train/data.json')))
data['probes'] = json.load(open(os.path.join(HERE, 'train/probes.json')))['probes']
st = data['stats']
for k, v in {'vocab': f"{st['corpus']['vocab']:,}", 'dim': str(st['corpus']['dim'])}.items():
    src = src.replace('{{' + k + '}}', v)
assert '{{' not in src

# hand-written claims in index.src.html, checked against the data
summ = json.load(open(os.path.join(HERE, 'train/summary.json')))
e3, loc = st['pca']['ends'], st['pca']['loc']
nn = {w: [x[0] for x in l] for w, l in summ['neighbours'].items()}
assert {'tybalt', 'juliet', 'mercutio'} <= set(nn['romeo']), 'romeo sits by tybalt, juliet, mercutio'
g = st['pca']['group']
tied = [('low', 0, 'The Taming of the Shrew'), ('high', 2, 'Coriolanus'), ('high', 1, '3 Henry VI'), ('low', 2, 'Romeo and Juliet')]
mixed = [('high', 0), ('low', 1)]
assert {'pompey', 'petruchio', 'gremio'} <= set(e3[0]['low'][:5]), 'axis 1: Shrew / Measure names'
assert {'tribunes', 'patricians', 'volsces'} <= set(e3[2]['high'][:5]), 'axis 3: Coriolanus'
assert {'george', 'london', 'norfolk'} <= set(e3[1]['high'][:5]), 'axis 2: English history plays'
for end, k, play in tied:   # "follow the plays": the group's busiest tenth beats 99% of random groups, inside that play
    eg = e3[k][end + '_group']
    assert eg['share'] > g['random_p99'] and eg['play'] == play, (k, end, eg)
for end, k in mixed:        # "only two ends look mixed": below the random groups' 95th percentile
    assert e3[k][end + '_group']['share'] < g['random_p95'], (k, end)
# run history sentence: the third entry comes from this run
assert (summ['analogy_hits'], next(a['rank'] for a in summ['analogies'] if a['q'][1] == 'king' and a['q'][2] == 'woman')) == (2, 28), 'update PRIOR / the run-history sentence'
assert 1 <= summ['analogy_hits'] < summ['analogy_total'] / 2, '"mostly fails"'
assert any(a['q'][1] == 'king' and a['q'][2] == 'woman' and not a['hit'] for a in st['analogies'])
assert set(['king', 'love', 'night', 'romeo', 'thou', 'sword']) <= set(data['probes']), 'chips are pre-registered probes'
blob = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
out = src.replace('/*DATA*/', blob)
open(os.path.join(HERE, 'index.html'), 'w').write(out)
print('index.html', os.path.getsize(os.path.join(HERE, 'index.html')), 'bytes')
