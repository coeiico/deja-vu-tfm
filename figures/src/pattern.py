"""
Draw three heads' attention on one test sequence for pattern.tex: build/pattern-<role>.tikz.

One cell per (query, key) pair, shaded by the attention weight; the keys a query cannot see
(later positions, removed by the causal mask) are back-hatched as one region; rules mark
where the two copies of the repeated segment begin; and for each query of the second copy, a
spot ring marks the key that head's role predicts: the previous token, the earlier copy of
the current token, or the token after that earlier copy. Layout only; the numbers are
data/pattern-default-seed0.json.
"""

import json

with open('data/pattern-default-seed0.json') as f:
  d = json.load(f)
size, period = len(d['tokens']), d['period']
second = size - period  # where the second copy starts
first = second - period
OFFSET = {'previous-token': 1, 'duplicate-token': period, 'induction': period - 1}

for role, h in d['heads'].items():
  pattern = h['pattern']
  out = []
  # The causal mask: everything above the diagonal, as one staircase polygon.
  stairs = ' -- '.join(f'({q + 1},{size - q}) -- ({q + 1},{size - q - 1})' for q in range(size - 1))
  out.append(f'\\fill[hatch back=ink!22] (1,{size}) -- {stairs} -- ({size},{size}) -- cycle;')
  for q in range(size):
    y = size - 1 - q
    for k in range(q + 1):
      shade = round(100 * min(1.0, pattern[q][k]))
      if shade >= 4:
        out.append(f'\\fill[ink!{shade}] ({k},{y}) rectangle ++(1,1);')
  for x in (first, second):
    out.append(f'\\draw[soft, line width=0.3pt] ({x},0) -- ({x},{size}) (0,{size - x}) -- ({size},{size - x});')
  out.append(f'\\draw[mframe] (0,0) rectangle ({size},{size});')
  for q in range(second, size):
    k = q - OFFSET[role]
    out.append(f'\\draw[spot, line width=0.35pt] ({k - 0.35},{size - 1 - q - 0.35}) rectangle ++(1.7,1.7);')
  with open(f'build/pattern-{role}.tikz', 'w') as f:
    f.write('\n'.join(out) + '\n')

names = {role: f'{h["layer"]}.{h["head"]}' for role, h in d['heads'].items()}
with open('build/pattern-values.tex', 'w') as f:
  f.write(
    f'\\def\\patternsize{{{size}}}\\def\\patternperiod{{{period}}}\\def\\patternfirst{{{first}}}\\def\\patternsecond{{{second}}}'
    f'\\def\\headPrev{{{names["previous-token"]}}}\\def\\headDup{{{names["duplicate-token"]}}}\\def\\headInd{{{names["induction"]}}}\n'
  )
