# Assignment 1 - Mini libreria per sistemi lineari

Implementazione Python dei quattro metodi iterativi richiesti dal progetto:

- Jacobi
- Gauss-Seidel
- Gradiente
- Gradiente coniugato

La matrice viene letta da file MatrixMarket `.mtx`, il vettore esatto e'
`x = ones(n)`, il termine noto e' `b = A x`, e ogni metodo parte da `x0 = 0`.
SciPy viene usato solo per leggere i file `.mtx`, conservare le matrici in
formato sparso CSR e fare i prodotti matrice-vettore. I quattro metodi iterativi
sono implementati direttamente nel file `solvers.py`. La sola sweep di
Gauss-Seidel (sostituzione in avanti, intrinsecamente sequenziale) e' compilata
con `numba` per renderla veloce, restando comunque codice nostro.

Il criterio di arresto e' quello del testo:

```text
||A x_k - b|| / ||b|| < tol
```

con `max_iter = 20000` di default.

## Dipendenze

```bash
python -m pip install -r requirements.txt
```

## Esecuzione completa

Dalla cartella del progetto:

```bash
python run_assignment.py
```

Questo comando usa tutti i file `.mtx` presenti nella cartella `dati` e le
tolleranze `1e-4`, `1e-6`, `1e-8`, `1e-10`.

Gli output vengono salvati in:

- `results/results.csv`
- `results/plots/relative_error.png`
- `results/plots/iterations.png`
- `results/plots/time.png`

## Esecuzione su una matrice o tolleranza specifica

```bash
python run_assignment.py --matrix dati/vem1.mtx --tol 1e-4
```

Si possono indicare piu' matrici o piu' tolleranze ripetendo gli argomenti:

```bash
python run_assignment.py --matrix dati/vem1.mtx --matrix dati/vem2.mtx --tol 1e-4 --tol 1e-6
```

Per generare solo il CSV senza grafici:

```bash
python run_assignment.py --no-plots
```

Per usare un solo metodo:

```bash
python run_assignment.py --method conjugate-gradient
```

I metodi disponibili sono `jacobi`, `gauss-seidel`, `gradient` e
`conjugate-gradient`. Si possono indicare piu' metodi ripetendo l'argomento.

## Test veloci

```bash
python test_assignment.py
```

I test controllano parser MatrixMarket, convergenza su matrice tridiagonale
tramite SciPy, convergenza su matrice tridiagonale SPD, e gradiente coniugato
in al piu' `n` iterazioni su un sistema piccolo.

## Numeri di condizionamento

```bash
python analyze_cond.py
```

Stima `lambda_min`, `lambda_max` e `cond(A)` per ogni matrice in `dati/`
(serve solo ai commenti della relazione, non fa parte dei solutori).

## Documenti

- `Teoria.md` — la teoria dietro i quattro metodi, spiegata in modo semplice
  (criterio di arresto, splitting, interpretazione come minimizzazione,
  zig-zag del gradiente, direzioni A-coniugate).
- `Relazione.md` — struttura della libreria, tabelle dei risultati, grafici e
  commenti.
